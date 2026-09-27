"""
features.py
Feature extraction pipeline for the Phishing URL Detector.

Tier 1 features are pure string/URL parsing - fast, deterministic, no network calls.
Tier 2 features require external lookups (WHOIS domain age, SSL info) and are
best-effort: if the lookup fails or is blocked (e.g. sandboxed / offline environment),
they fall back to neutral values so the pipeline never crashes.
"""
import re
import socket
from urllib.parse import urlparse

# Common URL-shortening services
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bl.ink", "shorte.st", "cutt.ly", "rebrand.ly", "tiny.cc",
    "rb.gy", "shorturl.at", "s.id",
}

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "bank", "update", "secure", "account", "confirm",
    "signin", "sign-in", "password", "webscr", "ebayisapi", "paypal",
    "billing", "suspend", "unlock", "alert", "support", "wallet",
    "recover", "invoice",
]

# Brand -> the domain(s) that legitimately belong to it. Used to catch the
# classic pattern of a brand name appearing in the URL (path, subdomain, or
# query) while the actual hostname is unrelated - e.g.
# "dghjdgf.com/paypal.co.uk/cycgi-bin/webscrcmd=..." references PayPal but
# is not hosted on paypal.com.
BRAND_DOMAINS = {
    "paypal": ["paypal.com"],
    "amazon": ["amazon.com", "amazon.co.uk", "amazon.de"],
    "apple": ["apple.com"],
    "microsoft": ["microsoft.com", "live.com", "office.com"],
    "netflix": ["netflix.com"],
    "ebay": ["ebay.com", "ebay.co.uk"],
    "facebook": ["facebook.com", "fb.com"],
    "instagram": ["instagram.com"],
    "google": ["google.com"],
    "irs": ["irs.gov"],
    "usps": ["usps.com"],
    "dhl": ["dhl.com"],
    "fedex": ["fedex.com"],
    "chase": ["chase.com"],
    "wellsfargo": ["wellsfargo.com"],
    "bankofamerica": ["bankofamerica.com"],
    "coinbase": ["coinbase.com"],
    "binance": ["binance.com"],
    "linkedin": ["linkedin.com"],
    "adobe": ["adobe.com"],
    "dropbox": ["dropbox.com"],
}

IP_REGEX = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$"  # IPv4
)

# Brand names used for typosquat / lookalike-domain checks (superset of
# BRAND_DOMAINS keys plus a few extra common typosquat targets).
TYPOSQUAT_BRANDS = list(BRAND_DOMAINS.keys())

# Common leetspeak / character-substitution tricks used in typosquats
_LEET_MAP = str.maketrans({
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a",
})

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "num_dots", "num_hyphens",
    "num_digits", "num_special_chars", "num_at", "num_double_slash_in_path",
    "has_ip_host", "is_shortened", "num_subdomains", "has_https",
    "suspicious_keyword_count", "num_query_params", "brand_impersonation",
    "typosquat_flag", "idn_homograph", "domain_age_days", "ssl_cert_ok",
]


def _safe_hostname(url: str) -> str:
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        return parsed.hostname or ""
    except Exception:
        return ""


def _levenshtein(a: str, b: str) -> int:
    """Standard edit-distance DP, stdlib only (no extra dependency)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def matched_typosquat(url: str):
    """Detects domains that closely resemble a known brand without being
    an exact match - e.g. 'paypa1.com', 'micr0soft-support.net',
    'gooogle.com'. Skips hostnames that genuinely belong to the brand.
    Returns a list of {brand, matched_token, reason} dicts."""
    hostname = _safe_hostname(url).lower()
    if not hostname:
        return []

    candidates = set()
    for label in hostname.split("."):
        candidates.add(label)
        for part in label.split("-"):
            if part:
                candidates.add(part)

    matches = []
    seen_brands = set()
    for brand in TYPOSQUAT_BRANDS:
        if brand in seen_brands:
            continue
        legit_domains = BRAND_DOMAINS.get(brand, [brand + ".com"])
        belongs = any(hostname == d or hostname.endswith("." + d) for d in legit_domains)
        if belongs:
            continue
        for cand in candidates:
            if not cand or cand == brand or len(cand) < 4:
                continue
            normalized = cand.translate(_LEET_MAP)
            if normalized == brand:
                matches.append({"brand": brand, "matched_token": cand, "reason": "character substitution"})
                seen_brands.add(brand)
                break
            dist = _levenshtein(cand, brand)
            if 1 <= dist <= 2:
                matches.append({"brand": brand, "matched_token": cand, "reason": "similar spelling"})
                seen_brands.add(brand)
                break
    return matches


def has_idn_homograph(url: str) -> int:
    """Flags punycode-encoded labels (xn--...) or raw non-ASCII characters
    in the hostname - both are used to visually spoof brand domains with
    look-alike Unicode characters (e.g. Cyrillic 'a')."""
    hostname = _safe_hostname(url)
    if not hostname:
        return 0
    if any(label.startswith("xn--") for label in hostname.split(".")):
        return 1
    try:
        hostname.encode("ascii")
    except UnicodeEncodeError:
        return 1
    return 0


def matched_brand_impersonation(url: str):
    """Returns list of brand names that appear in the URL while the
    hostname does not actually belong to that brand - the classic
    'legit-looking brand name, unrelated real domain' phishing pattern."""
    hostname = _safe_hostname(url).lower()
    low = url.lower()
    matches = []
    for brand, legit_domains in BRAND_DOMAINS.items():
        if brand in low:
            belongs = any(
                hostname == d or hostname.endswith("." + d) for d in legit_domains
            )
            if not belongs:
                matches.append(brand)
    return matches


def extract_tier1_features(url: str) -> dict:
    """Fast, deterministic, no network access. Always available."""
    url = url.strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    special_chars = sum(url.count(c) for c in ["%", "=", "&", "_", "~", "$", "+", "*", "!"])

    subdomain_count = 0
    if hostname:
        parts = hostname.split(".")
        # crude heuristic: parts minus domain+TLD
        subdomain_count = max(0, len(parts) - 2)

    keyword_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url.lower())
    brand_impersonation = 1 if matched_brand_impersonation(url) else 0
    typosquat_flag = 1 if matched_typosquat(url) else 0
    idn_homograph = has_idn_homograph(url)

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": special_chars,
        "num_at": url.count("@"),
        "num_double_slash_in_path": path.count("//"),
        "has_ip_host": 1 if IP_REGEX.match(hostname) else 0,
        "is_shortened": 1 if hostname.lower() in SHORTENERS else 0,
        "num_subdomains": subdomain_count,
        "has_https": 1 if parsed.scheme == "https" else 0,
        "suspicious_keyword_count": keyword_count,
        "num_query_params": len(query.split("&")) if query else 0,
        "brand_impersonation": brand_impersonation,
        "typosquat_flag": typosquat_flag,
        "idn_homograph": idn_homograph,
    }


def extract_tier2_features(url: str, timeout: float = 2.0) -> dict:
    """
    Best-effort external lookups. Falls back to neutral/unknown values
    (domain_age_days = -1, ssl_cert_ok = -1) if the network call fails or
    is unavailable (e.g. blocked egress in a sandboxed environment).
    Wire in `python-whois` / a real SSL check in production.
    """
    hostname = _safe_hostname(url)
    domain_age_days = -1
    ssl_cert_ok = -1

    if hostname:
        try:
            import whois  # python-whois, optional dependency
            w = whois.whois(hostname)
            creation = w.creation_date
            if isinstance(creation, list):
                creation = creation[0]
            if creation:
                from datetime import datetime
                domain_age_days = (datetime.now() - creation).days
        except Exception:
            domain_age_days = -1

        try:
            import ssl
            ctx = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=timeout) as sock:
                with ctx.wrap_socket(sock, server_hostname=hostname):
                    ssl_cert_ok = 1
        except Exception:
            ssl_cert_ok = 0

    return {"domain_age_days": domain_age_days, "ssl_cert_ok": ssl_cert_ok}


def extract_features(url: str, use_tier2: bool = True) -> dict:
    feats = extract_tier1_features(url)
    if use_tier2:
        feats.update(extract_tier2_features(url))
    else:
        feats.update({"domain_age_days": -1, "ssl_cert_ok": -1})
    # ensure consistent key order
    return {k: feats.get(k, 0) for k in FEATURE_NAMES}


def matched_suspicious_keywords(url: str):
    low = url.lower()
    return [kw for kw in SUSPICIOUS_KEYWORDS if kw in low]
