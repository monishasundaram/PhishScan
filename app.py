"""
app.py - Flask backend for the AI-Based Phishing URL Detection & Risk Analysis System.
"""
import json
import os
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import shap
from flask import Flask, request, jsonify, render_template, g

from features import (
    extract_features, extract_tier1_features, matched_suspicious_keywords,
    matched_brand_impersonation, matched_typosquat, FEATURE_NAMES,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "data", "scans.db")
MODEL_PATH = os.path.join(APP_DIR, "data", "model.joblib")

app = Flask(__name__)

# ---------------------------------------------------------------- model ----
_bundle = joblib.load(MODEL_PATH)
MODEL = _bundle["model"]
MODEL_NAME = _bundle["model_name"]
THRESHOLD = _bundle["threshold"]
_BACKGROUND = _bundle.get("background")

# Human-readable labels for the SHAP feature-contribution panel.
FEATURE_LABELS = {
    "url_length": "URL length", "hostname_length": "Hostname length",
    "path_length": "Path length", "num_dots": "Number of dots",
    "num_hyphens": "Number of hyphens", "num_digits": "Number of digits",
    "num_special_chars": "Special characters", "num_at": "'@' symbol present",
    "num_double_slash_in_path": "Double slash in path", "has_ip_host": "Raw IP as host",
    "is_shortened": "URL shortener used", "num_subdomains": "Subdomain count",
    "has_https": "HTTPS present", "suspicious_keyword_count": "Suspicious keywords",
    "num_query_params": "Query parameter count", "brand_impersonation": "Brand impersonation",
    "typosquat_flag": "Typosquat pattern", "idn_homograph": "Non-ASCII/punycode domain",
    "domain_age_days": "Domain age (days)", "ssl_cert_ok": "Valid SSL certificate",
}


def _build_explainer():
    """Builds a SHAP explainer matched to the production model type.
    Returns None if SHAP can't be set up (e.g. no background sample saved
    by an older model.joblib) - callers must handle that gracefully."""
    if _BACKGROUND is None:
        return None
    try:
        if MODEL_NAME in ("random_forest", "xgboost"):
            return shap.TreeExplainer(
                MODEL, _BACKGROUND, model_output="probability",
                feature_perturbation="interventional",
            )
        masker = shap.maskers.Independent(_BACKGROUND)
        return shap.LinearExplainer(MODEL, masker)
    except Exception as e:
        print(f"WARNING: could not build SHAP explainer: {e}")
        return None


EXPLAINER = _build_explainer()


def get_shap_breakdown(feats: dict, top_k: int = 5):
    """Returns the top_k features that most influenced this specific
    prediction, as [{feature, label, value, contribution_pct, direction}],
    ranked by absolute SHAP magnitude. Falls back to [] if SHAP is
    unavailable rather than failing the request."""
    if EXPLAINER is None:
        return []
    try:
        X = np.array([[feats[f] for f in FEATURE_NAMES]])
        if MODEL_NAME in ("random_forest", "xgboost"):
            raw = EXPLAINER.shap_values(X, check_additivity=False)
            # shape (1, n_features, n_classes) -> take the phishing class
            values = np.asarray(raw)[0, :, 1] if np.ndim(raw) == 3 else np.asarray(raw)[0]
        else:
            values = EXPLAINER(X).values[0]

        total_abs = np.sum(np.abs(values)) or 1.0
        order = np.argsort(-np.abs(values))[:top_k]
        breakdown = []
        for idx in order:
            fname = FEATURE_NAMES[idx]
            sv = float(values[idx])
            if abs(sv) < 1e-6:
                continue
            breakdown.append({
                "feature": fname,
                "label": FEATURE_LABELS.get(fname, fname),
                "value": feats[fname],
                "contribution_pct": round(100 * abs(sv) / total_abs, 1),
                "direction": "increases_risk" if sv > 0 else "decreases_risk",
            })
        return breakdown
    except Exception as e:
        print(f"WARNING: SHAP computation failed: {e}")
        return []

# Feature-level thresholds used purely for the human-readable "reasons" panel
# (independent of the ML decision itself, which uses the full feature vector).
REASON_RULES = [
    ("has_ip_host", lambda v: v == 1, "URL uses a raw IP address instead of a domain name"),
    ("is_shortened", lambda v: v == 1, "URL uses a link-shortening service"),
    ("has_https", lambda v: v == 0, "Connection is not secured with HTTPS"),
    ("num_subdomains", lambda v: v >= 3, "Unusually high number of subdomains"),
    ("num_hyphens", lambda v: v >= 3, "Unusually high number of hyphens in the URL"),
    ("url_length", lambda v: v >= 75, "URL is unusually long"),
    ("num_at", lambda v: v >= 1, "URL contains '@', a common redirection trick"),
    ("domain_age_days", lambda v: 0 <= v < 30, "Domain was registered very recently"),
    ("ssl_cert_ok", lambda v: v == 0, "Could not verify a valid SSL certificate"),
    ("idn_homograph", lambda v: v == 1, "Domain uses encoded/non-ASCII characters that can visually spoof a brand"),
]


# --------------------------------------------------------------------- db ----
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            prediction TEXT CHECK(prediction IN ('phishing','legitimate')),
            risk_score REAL,
            model_version TEXT,
            features_json TEXT,
            top_reasons_json TEXT
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER REFERENCES scans(id),
            user_label TEXT CHECK(user_label IN ('phishing','legitimate')),
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()


# ------------------------------------------------------------ prediction ----
def predict_url(url: str, use_tier2: bool = True):
    feats = extract_features(url, use_tier2=use_tier2)
    X = np.array([[feats[f] for f in FEATURE_NAMES]])
    proba = float(MODEL.predict_proba(X)[0, 1])

    # High-confidence rule-based overrides: typosquatted domains and raw-IP
    # hosts are near-zero-false-positive signals (no legitimate business
    # uses paypa1.com or a bare IP address as its real domain). Since this
    # project prioritizes recall over precision, these force a phishing
    # verdict regardless of what the ML model alone scores.
    forced_phishing = feats.get("typosquat_flag") == 1 or feats.get("has_ip_host") == 1
    if forced_phishing:
        proba = max(proba, 0.99)
        label = "phishing"
    else:
        label = "phishing" if proba >= THRESHOLD else "legitimate"

    reasons = []
    for fname, cond, msg in REASON_RULES:
        try:
            if cond(feats[fname]):
                reasons.append(msg)
        except Exception:
            pass
    for brand in matched_brand_impersonation(url):
        reasons.append(f"References brand \"{brand}\" but is not hosted on {brand}'s real domain")
    for m in matched_typosquat(url):
        reasons.append(
            f"Domain \"{m['matched_token']}\" closely resembles brand \"{m['brand']}\" "
            f"({m['reason']}) but isn't {m['brand']}'s real domain"
        )
    for kw in matched_suspicious_keywords(url):
        reasons.append(f"Suspicious keyword detected: \"{kw}\"")

    if not reasons:
        reasons.append("No individually suspicious signals; verdict based on overall feature pattern")

    shap_breakdown = get_shap_breakdown(feats)

    return {
        "url": url,
        "prediction": label,
        "risk_score": round(proba * 100, 1),
        "model": MODEL_NAME,
        "threshold": round(THRESHOLD, 3),
        "reasons": reasons[:8],
        "shap_explanation": shap_breakdown,
        "features": feats,
    }


# --------------------------------------------------------------- routes ----
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/v1/scan", methods=["POST"])
def scan():
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Missing 'url' in request body"}), 400
    if len(url) > 2048:
        return jsonify({"error": "URL too long"}), 400

    # Tier-2 (WHOIS/SSL) lookups are best-effort and may be skipped for speed
    use_tier2 = bool(payload.get("deep_scan", True))
    result = predict_url(url, use_tier2=use_tier2)

    db = get_db()
    cur = db.execute(
        """INSERT INTO scans (url, prediction, risk_score, model_version, features_json, top_reasons_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            result["url"], result["prediction"], result["risk_score"], result["model"],
            json.dumps(result["features"]), json.dumps(result["reasons"]),
        ),
    )
    db.commit()
    result["scan_id"] = cur.lastrowid
    return jsonify(result)


@app.route("/api/v1/scans", methods=["GET"])
def list_scans():
    limit = min(int(request.args.get("limit", 50)), 500)
    db = get_db()
    rows = db.execute(
        "SELECT id, url, submitted_at, prediction, risk_score FROM scans ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/v1/stats/timeseries", methods=["GET"])
def stats_timeseries():
    db = get_db()
    rows = db.execute(
        """SELECT date(submitted_at) AS day,
                  SUM(CASE WHEN prediction='phishing' THEN 1 ELSE 0 END) AS phishing,
                  SUM(CASE WHEN prediction='legitimate' THEN 1 ELSE 0 END) AS legitimate
           FROM scans GROUP BY day ORDER BY day ASC"""
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/v1/feedback", methods=["POST"])
def feedback():
    payload = request.get_json(silent=True) or {}
    scan_id = payload.get("scan_id")
    user_label = payload.get("user_label")
    if user_label not in ("phishing", "legitimate") or not scan_id:
        return jsonify({"error": "scan_id and user_label ('phishing'|'legitimate') are required"}), 400
    db = get_db()
    db.execute(
        "INSERT INTO feedback (scan_id, user_label) VALUES (?, ?)", (scan_id, user_label)
    )
    db.commit()
    return jsonify({"status": "recorded"})


@app.route("/api/v1/stats/summary", methods=["GET"])
def stats_summary():
    db = get_db()
    row = db.execute(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN prediction='phishing' THEN 1 ELSE 0 END) AS phishing,
                  SUM(CASE WHEN prediction='legitimate' THEN 1 ELSE 0 END) AS legitimate,
                  AVG(risk_score) AS avg_risk
           FROM scans"""
    ).fetchone()
    feedback_count = db.execute("SELECT COUNT(*) AS c FROM feedback").fetchone()["c"]
    return jsonify({
        "total": row["total"] or 0,
        "phishing": row["phishing"] or 0,
        "legitimate": row["legitimate"] or 0,
        "avg_risk": round(row["avg_risk"] or 0, 1),
        "feedback": feedback_count,
    })


@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "threshold": round(THRESHOLD, 3)})


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
else:
    init_db()
