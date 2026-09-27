"""
train_model.py

Builds a labeled URL dataset, extracts Tier-1 features (network-independent,
so training works identically offline or online), trains and compares
Logistic Regression / Random Forest / XGBoost, tunes the decision threshold
to hit a recall floor, and saves the winning model + threshold to disk.

NOTE ON DATA: By default this script uses a programmatically generated demo
dataset (real legitimate domains + realistically-patterned synthetic
phishing URLs) so the project runs end-to-end without external downloads.

To train on REAL data instead, pass CSV files:

    python train_model.py --phishing-csv data/real_phishing.csv --legit-csv data/real_legit.csv

- --phishing-csv: a PhishTank or OpenPhish export. Any CSV/TXT with a 'url'
  column (or one URL per line) works.
- --legit-csv: a Tranco top-sites export (columns: rank,domain). Each domain
  is turned into a few realistic URLs.

The hand-crafted "hard" examples (typosquats, brand-in-path, real-brand
login pages, brand-mention-on-news-site) are always added on top of
whichever base dataset you choose, real or synthetic, since they target
specific known model weaknesses regardless of data source.
"""
import argparse
import csv
import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, classification_report, confusion_matrix,
    recall_score, precision_score, f1_score, fbeta_score, roc_auc_score,
)
from xgboost import XGBClassifier

from features import extract_tier1_features, FEATURE_NAMES, SUSPICIOUS_KEYWORDS, BRAND_DOMAINS

random.seed(42)
np.random.seed(42)

LEGIT_DOMAINS = [
    "google.com", "wikipedia.org", "github.com", "amazon.com", "microsoft.com",
    "apple.com", "nytimes.com", "bbc.com", "linkedin.com", "reddit.com",
    "stackoverflow.com", "spotify.com", "netflix.com", "dropbox.com",
    "salesforce.com", "adobe.com", "cloudflare.com", "shopify.com",
    "airbnb.com", "wordpress.com", "medium.com", "twitch.tv", "espn.com",
    "cnn.com", "forbes.com", "nike.com", "target.com", "walmart.com",
    "chase.com", "wellsfargo.com", "paypal.com", "ebay.com", "yahoo.com",
    "bing.com", "twitter.com", "instagram.com", "pinterest.com", "zoom.us",
    "slack.com", "notion.so", "figma.com", "canva.com", "trello.com",
    "irs.gov", "usps.com", "fedex.com", "ups.com", "coursera.org",
    "khanacademy.org", "mit.edu", "stanford.edu", "who.int",
]

LEGIT_PATHS = [
    "", "/", "/about", "/products", "/blog/2024/annual-report",
    "/account/settings", "/login", "/support/contact-us", "/careers",
    "/search?q=example", "/docs/api/v2", "/news/technology",
    "/user/profile?id=8842", "/checkout/cart", "/help/faq",
]

BRAND_LOOKALIKES = [
    "paypal", "amazon", "apple", "microsoft", "netflix", "bankofamerica",
    "wellsfargo", "chase", "facebook", "instagram", "google", "irs",
    "dhl", "usps", "ebay", "coinbase", "binance",
]

PHISHY_TLDS = [".tk", ".ml", ".ga", ".cf", ".xyz", ".top", ".click", ".gq"]


def random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def gen_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(LEGIT_PATHS)
    scheme = "https"
    sub = random.choice(["", "www.", "app.", "shop."])
    return f"{scheme}://{sub}{domain}{path}"


def gen_phishing_url():
    brand = random.choice(BRAND_LOOKALIKES)
    kw = random.choice(SUSPICIOUS_KEYWORDS)
    style = random.randint(0, 5)

    if style == 0:
        # raw IP host
        return f"http://{random_ip()}/{brand}/{kw}.php"
    if style == 1:
        # brand as subdomain of unrelated domain, hyphenated, suspicious kw
        rand_domain = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(5, 10)))
        return f"http://{brand}-{kw}.{rand_domain}{random.choice(PHISHY_TLDS)}/account/{kw}"
    if style == 2:
        # excessive subdomains
        return f"http://{kw}.{brand}.secure.verify.{random.choice(['info','biz','xyz'])}/login.html"
    if style == 3:
        # @ trick
        rand_domain = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=8))
        return f"http://{brand}.com@{rand_domain}.ru/{kw}"
    if style == 4:
        # shortener-style token
        token = "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFG0123456789", k=7))
        return f"http://bit.ly/{token}"
    # long, hyphen and digit-heavy, no https
    rand_domain = "-".join(random.choices(
        [brand, kw, "secure", "update", str(random.randint(100, 999))], k=4))
    return f"http://{rand_domain}.{random.choice(PHISHY_TLDS)}/{kw}-{random.randint(1000,9999)}"


def gen_hard_legit_url():
    """Legitimate URLs that intentionally look a bit suspicious (long,
    tracking params, a suspicious keyword in path) to avoid a trivially
    separable synthetic dataset."""
    domain = random.choice(LEGIT_DOMAINS)
    kw = random.choice(SUSPICIOUS_KEYWORDS)
    return f"https://www.{domain}/account/{kw}?ref=email&utm_id={random.randint(10000,99999)}&token={random.randint(100000,999999)}"


def gen_hard_legit_brand_mention_url():
    """A legitimate news/blog domain that happens to mention a brand name
    in the path (e.g. an article about PayPal) - NOT hosted on that brand's
    domain, but also not phishing. Keeps brand_impersonation from becoming
    a trivial shortcut feature."""
    news_domains = ["nytimes.com", "bbc.com", "forbes.com", "techcrunch.com",
                     "reuters.com", "cnn.com", "theverge.com"]
    brand = random.choice(list(BRAND_DOMAINS.keys()))
    domain = random.choice(news_domains)
    slug = random.choice([
        f"{brand}-quarterly-earnings-report", f"why-{brand}-stock-is-rising",
        f"{brand}-security-update-explained", f"review-of-{brand}-new-app",
    ])
    return f"https://www.{domain}/business/2026/{slug}"


def gen_real_brand_login_url():
    """Real brand domains with the ordinary account/security-related paths
    they legitimately have (login, verify, account, secure). Without enough
    of these, a model over-learns 'suspicious keyword present' as a phishing
    signal even when the hostname genuinely IS the brand - exactly the false
    positive seen on https://www.paypal.com/signin during testing."""
    brand_domains = [d for domains in BRAND_DOMAINS.values() for d in domains]
    domain = random.choice(brand_domains)
    path = random.choice([
        "/login", "/signin", "/account", "/account/security",
        "/account/verify", "/secure/update-info", "/wallet",
        "/billing/invoice", "/support", "/recover-password",
    ])
    return f"https://www.{domain}{path}"


def gen_phishing_url_typosquat():
    """Typosquat / lookalike-domain phishing: character substitution
    (paypa1.com, micr0soft.net) or near-miss spelling (gooogle.com,
    amazom.com) of a real brand domain."""
    brand = random.choice(list(BRAND_DOMAINS.keys()))
    style = random.randint(0, 2)
    if style == 0:
        # leetspeak substitution
        subs = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5", "t": "7"}
        chars = list(brand)
        idx = random.randrange(len(chars))
        if chars[idx] in subs:
            chars[idx] = subs[chars[idx]]
        domain = "".join(chars)
    elif style == 1:
        # doubled letter / dropped letter (edit distance 1)
        idx = random.randrange(1, len(brand))
        domain = brand[:idx] + brand[idx - 1] + brand[idx:]  # doubled letter
    else:
        # adjacent-letter swap
        idx = random.randrange(0, len(brand) - 1)
        chars = list(brand)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        domain = "".join(chars)
    tld = random.choice([".com", ".net", ".info", ".online", ".site"])
    path = random.choice(["/login", "/secure/verify", "/account/update", "/signin", ""])
    scheme = random.choice(["http", "https"])
    return f"{scheme}://www.{domain}{tld}{path}"


def gen_phishing_url_brand_in_path():
    """Real-world pattern: brand name stuffed into the PATH of an unrelated,
    often gibberish, domain - e.g. dghjdgf.com/paypal.co.uk/cycgi-bin/
    webscrcmd=_home-customer&nav=1/loading.php. This is distinct from
    gen_phishing_url()'s brand-in-domain/subdomain patterns and is what
    closes the gap that let such URLs slip through as 'legitimate'."""
    brand = random.choice(list(BRAND_DOMAINS.keys()))
    fake_domain = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(6, 10)))
    tld = random.choice([".com", ".net", ".info", ".xyz", ".ru", ".top"])
    kit_file = random.choice([
        "cycgi-bin/webscrcmd=_home-customer&nav=1/loading.php",
        "cgi-bin/webscr?cmd=_login-run",
        "secure/update-account.php",
        "myaccount/signin/challenge",
        "confirm-identity/step2.html",
    ])
    brand_path = random.choice([brand, f"{brand}.com", f"{brand}.co.uk", f"{brand}-secure"])
    scheme = random.choice(["http", "https"])  # phishing sites increasingly use HTTPS too
    return f"{scheme}://www.{fake_domain}{tld}/{brand_path}/{kit_file}"


def build_dataset(n_legit=1400, n_phish=900, n_hard_legit=150,
                   n_brand_path_phish=350, n_brand_mention_legit=150,
                   n_typosquat_phish=350, n_real_brand_login=250):
    rows = []
    for _ in range(n_legit):
        rows.append({"url": gen_legit_url(), "label": 0})
    for _ in range(n_hard_legit):
        rows.append({"url": gen_hard_legit_url(), "label": 0})
    for _ in range(n_brand_mention_legit):
        rows.append({"url": gen_hard_legit_brand_mention_url(), "label": 0})
    for _ in range(n_real_brand_login):
        rows.append({"url": gen_real_brand_login_url(), "label": 0})
    for _ in range(n_phish):
        rows.append({"url": gen_phishing_url(), "label": 1})
    for _ in range(n_brand_path_phish):
        rows.append({"url": gen_phishing_url_brand_in_path(), "label": 1})
    for _ in range(n_typosquat_phish):
        rows.append({"url": gen_phishing_url_typosquat(), "label": 1})
    df = pd.DataFrame(rows).drop_duplicates(subset="url").reset_index(drop=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def build_hard_examples(n_hard_legit=150, n_brand_mention_legit=150,
                         n_real_brand_login=250, n_brand_path_phish=350,
                         n_typosquat_phish=350):
    """The hand-crafted examples that target specific known model
    weaknesses (typosquats, brand-in-path phishing, real-brand login
    pages, brand-mentions-on-news-sites). Layered on top of either the
    synthetic base dataset or real PhishTank/Tranco data."""
    rows = []
    for _ in range(n_hard_legit):
        rows.append({"url": gen_hard_legit_url(), "label": 0})
    for _ in range(n_brand_mention_legit):
        rows.append({"url": gen_hard_legit_brand_mention_url(), "label": 0})
    for _ in range(n_real_brand_login):
        rows.append({"url": gen_real_brand_login_url(), "label": 0})
    for _ in range(n_brand_path_phish):
        rows.append({"url": gen_phishing_url_brand_in_path(), "label": 1})
    for _ in range(n_typosquat_phish):
        rows.append({"url": gen_phishing_url_typosquat(), "label": 1})
    return pd.DataFrame(rows)


def load_phishing_csv(path: str) -> list:
    """Loads phishing URLs from a PhishTank/OpenPhish export. Accepts a
    CSV with a 'url' column, a CSV with no header (first column = url),
    or a plain text file with one URL per line."""
    urls = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sniffer_sample = f.read(2048)
        f.seek(0)
        is_csv = "," in sniffer_sample and "\n" in sniffer_sample
        if is_csv:
            reader = csv.DictReader(f)
            if reader.fieldnames and "url" in [c.lower() for c in reader.fieldnames]:
                url_col = next(c for c in reader.fieldnames if c.lower() == "url")
                for row in reader:
                    u = row.get(url_col, "").strip()
                    if u:
                        urls.append(u)
            else:
                f.seek(0)
                for row in csv.reader(f):
                    if row and row[0].strip().lower() not in ("url", ""):
                        urls.append(row[0].strip())
        else:
            for line in f:
                line = line.strip()
                if line:
                    urls.append(line)
    return urls


def load_legit_csv(path: str, paths_per_domain: int = 2) -> list:
    """Loads a Tranco-style top-sites export (columns: rank,domain, no
    header) and expands each bare domain into a few realistic URLs using
    the same LEGIT_PATHS templates as the synthetic generator."""
    urls = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            domain = row[1].strip() if len(row) >= 2 else row[0].strip()
            if not domain or "." not in domain:
                continue
            for _ in range(paths_per_domain):
                path = random.choice(LEGIT_PATHS)
                sub = random.choice(["", "www."])
                urls.append(f"https://{sub}{domain}{path}")
    return urls


def build_dataset_from_real_data(phishing_csv: str, legit_csv: str,
                                  max_phishing: int = 5000, max_legit: int = 5000) -> pd.DataFrame:
    """Builds the training set from real PhishTank/OpenPhish + Tranco data,
    plus the hand-crafted hard examples layered on top."""
    print(f"Loading phishing URLs from {phishing_csv} ...")
    phish_urls = load_phishing_csv(phishing_csv)[:max_phishing]
    print(f"  loaded {len(phish_urls)} phishing URLs")

    print(f"Loading legitimate domains from {legit_csv} ...")
    legit_urls = load_legit_csv(legit_csv)[:max_legit]
    print(f"  loaded {len(legit_urls)} legitimate URLs")

    rows = [{"url": u, "label": 1} for u in phish_urls]
    rows += [{"url": u, "label": 0} for u in legit_urls]

    hard_df = build_hard_examples()
    df = pd.concat([pd.DataFrame(rows), hard_df], ignore_index=True)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    # Tier-1 only for training: deterministic + offline-safe. Tier-2
    # (domain age / ssl) is captured live at inference time in app.py.
    feats = df["url"].apply(extract_tier1_features).apply(pd.Series)
    feats["domain_age_days"] = -1
    feats["ssl_cert_ok"] = -1
    return feats[FEATURE_NAMES]


def pick_threshold_for_recall(y_true, y_proba, min_recall=0.95):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    best_t, best_p = 0.5, 0.0
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= min_recall and p >= best_p:
            best_p, best_t = p, t
    return best_t


def evaluate(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred, target_names=["legitimate", "phishing"]))
    print("Confusion matrix [[TN FP][FN TP]]:\n", confusion_matrix(y_true, y_pred))
    return {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f2": fbeta_score(y_true, y_pred, beta=2),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the phishing URL detector")
    parser.add_argument("--phishing-csv", help="Path to a PhishTank/OpenPhish export (real phishing URLs)")
    parser.add_argument("--legit-csv", help="Path to a Tranco top-sites export (real legitimate domains)")
    args = parser.parse_args()

    if args.phishing_csv and args.legit_csv:
        print(f"Using REAL data: {args.phishing_csv} + {args.legit_csv}")
        df = build_dataset_from_real_data(args.phishing_csv, args.legit_csv)
    else:
        print("No --phishing-csv/--legit-csv given, building synthetic demo dataset...")
        df = build_dataset()
    print(f"Total URLs: {len(df)}  (phishing={df.label.sum()}, legit={(df.label==0).sum()})")

    X = featurize(df)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    results = {}

    # --- Logistic Regression baseline ---
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_train, y_train)
    results["logreg"] = {
        "model": lr,
        "proba": lr.predict_proba(X_test)[:, 1],
    }

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results["random_forest"] = {
        "model": rf,
        "proba": rf.predict_proba(X_test)[:, 1],
    }

    # --- XGBoost (recall-tuned via scale_pos_weight) ---
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = (neg / pos) * 1.3  # push a bit further toward recall
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    results["xgboost"] = {
        "model": xgb,
        "proba": xgb.predict_proba(X_test)[:, 1],
    }

    # Evaluate all at default 0.5 threshold first, for comparison
    summary = {}
    for name, r in results.items():
        preds = (r["proba"] >= 0.5).astype(int)
        summary[name] = evaluate(name, y_test, preds)
        summary[name]["auc"] = roc_auc_score(y_test, r["proba"])

    print("\n=== Model comparison @ threshold 0.5 ===")
    print(pd.DataFrame(summary).T.round(3))

    # Pick production model = best F2 (recall-weighted) among the three
    best_name = max(summary, key=lambda k: summary[k]["f2"])
    print(f"\nSelected production model: {best_name}")

    best_model = results[best_name]["model"]
    best_proba = results[best_name]["proba"]

    threshold = pick_threshold_for_recall(y_test, best_proba, min_recall=0.95)
    print(f"Tuned decision threshold for recall>=0.95: {threshold:.3f}")

    tuned_preds = (best_proba >= threshold).astype(int)
    evaluate(f"{best_name} @ tuned threshold", y_test, tuned_preds)

    # Background sample for SHAP explainer (small, representative subset of
    # the training features - used to build the explainer at serve time).
    background = X_train.sample(n=min(100, len(X_train)), random_state=42)

    os.makedirs("data", exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "threshold": float(threshold),
            "feature_names": FEATURE_NAMES,
            "background": background,
        },
        "data/model.joblib",
    )
    df.to_csv("data/training_urls.csv", index=False)
    print("\nSaved model -> data/model.joblib")


if __name__ == "__main__":
    main()
