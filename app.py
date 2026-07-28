from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import os
import sqlite3
import re
import joblib
import numpy as np
import pandas as pd
import requests
import tldextract

from bs4 import BeautifulSoup
from feature_extractor import extract_all_features
from urllib.parse import urlparse
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO
from flask import make_response
from xhtml2pdf import pisa
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key_123")

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False   # Make True when using HTTPS
app.config["PERMANENT_SESSION_LIFETIME"] = 1800


# ---------------- LOAD MODEL ----------------
MODEL_PATH = "model/ensemble_pipeline_model.pkl"
FEATURE_NAMES_PATH = "model/feature_names.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Ensemble model loaded successfully.")
except Exception as e:
    model = None
    print("Error loading model:", e)

try:
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print("Feature names loaded successfully.")
except Exception as e:
    feature_names = None
    print("Feature names not loaded:", e)


# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            url TEXT,
            prediction TEXT,
            probability REAL,
            risk TEXT,
            scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
      CREATE TABLE IF NOT EXISTS contact_messages (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         name TEXT NOT NULL,
         email TEXT NOT NULL,
         subject TEXT NOT NULL,
         message TEXT NOT NULL,
         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ---------------- HELPER FUNCTIONS ----------------
def safe_div(a, b):
    return a / b if b != 0 else 0


def get_risk_level(probability):
    if probability >= 0.80:
        return "High Risk"
    elif probability >= 0.50:
        return "Suspicious"
    else:
        return "Safe"


def get_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def is_valid_url(url):
    try:
        parsed = urlparse(url)
        return parsed.scheme in ["http", "https"] and parsed.netloc != ""
    except Exception:
        return False


# ---------------- URL FEATURE EXTRACTION ----------------
def extract_url_features(url: str) -> dict:
    parsed = urlparse(url)
    full_url = url
    hostname = parsed.netloc
    path = parsed.path

    features = {}

    features["length_url"] = len(full_url)
    features["length_hostname"] = len(hostname)
    features["ip"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 0
    features["nb_dots"] = full_url.count(".")
    features["nb_hyphens"] = full_url.count("-")
    features["nb_at"] = full_url.count("@")
    features["nb_qm"] = full_url.count("?")
    features["nb_and"] = full_url.count("&")
    features["nb_or"] = full_url.count("|")
    features["nb_eq"] = full_url.count("=")
    features["nb_underscore"] = full_url.count("_")
    features["nb_tilde"] = full_url.count("~")
    features["nb_percent"] = full_url.count("%")
    features["nb_slash"] = full_url.count("/")
    features["nb_star"] = full_url.count("*")
    features["nb_colon"] = full_url.count(":")
    features["nb_comma"] = full_url.count(",")
    features["nb_semicolumn"] = full_url.count(";")
    features["nb_dollar"] = full_url.count("$")
    features["nb_space"] = full_url.count(" ")
    features["nb_www"] = full_url.lower().count("www")
    features["nb_com"] = full_url.lower().count(".com")
    features["nb_dslash"] = full_url.count("//") - 1 if "://" in full_url else full_url.count("//")

    features["http_in_path"] = 1 if "http" in path.lower() else 0
    features["https_token"] = 1 if "https" in hostname.lower().replace("https", "") else 0

    digits_url = sum(c.isdigit() for c in full_url)
    digits_host = sum(c.isdigit() for c in hostname)
    features["ratio_digits_url"] = safe_div(digits_url, len(full_url))
    features["ratio_digits_host"] = safe_div(digits_host, len(hostname))

    features["punycode"] = 1 if "xn--" in full_url.lower() else 0

    try:
        features["port"] = 1 if parsed.port else 0
    except ValueError:
        features["port"] = 0

    subdomains = hostname.split(".") if hostname else []
    features["tld_in_path"] = 0
    features["tld_in_subdomain"] = 0
    features["abnormal_subdomain"] = 1 if len(subdomains) > 3 else 0
    features["nb_subdomains"] = max(len(subdomains) - 2, 0)
    features["prefix_suffix"] = 1 if "-" in hostname else 0
    features["random_domain"] = 1 if re.search(r"[a-z0-9]{10,}", hostname.replace(".", "")) else 0

    shorteners = [
        "bit.ly", "goo.gl", "tinyurl", "ow.ly", "t.co",
        "is.gd", "buff.ly", "adf.ly", "bit.do", "cutt.ly"
    ]
    features["shortening_service"] = 1 if any(s in hostname.lower() for s in shorteners) else 0
    features["path_extension"] = 1 if re.search(r"\.[a-zA-Z0-9]+$", path) else 0
    features["nb_redirection"] = full_url.count("//")
    features["nb_external_redirection"] = 0

    words_raw = [w for w in re.split(r"[\/\.\-\_\?\=\&\:]+", full_url) if w]
    host_words = [w for w in re.split(r"[\.\\-]+", hostname) if w]
    path_words = [w for w in re.split(r"[\/\.\-\_\?\=\&\:]+", path) if w]

    features["length_words_raw"] = len(words_raw)
    features["char_repeat"] = 1 if re.search(r"(.)\1{2,}", full_url) else 0
    features["shortest_words_raw"] = min([len(w) for w in words_raw], default=0)
    features["shortest_word_host"] = min([len(w) for w in host_words], default=0)
    features["shortest_word_path"] = min([len(w) for w in path_words], default=0)
    features["longest_words_raw"] = max([len(w) for w in words_raw], default=0)
    features["longest_word_host"] = max([len(w) for w in host_words], default=0)
    features["longest_word_path"] = max([len(w) for w in path_words], default=0)
    features["avg_words_raw"] = float(np.mean([len(w) for w in words_raw])) if words_raw else 0
    features["avg_word_host"] = float(np.mean([len(w) for w in host_words])) if host_words else 0
    features["avg_word_path"] = float(np.mean([len(w) for w in path_words])) if path_words else 0

    phish_hints = ["login", "verify", "update", "secure", "account", "bank", "confirm", "password", "signin"]
    features["phish_hints"] = sum(1 for hint in phish_hints if hint in full_url.lower())

    features["domain_in_brand"] = 0
    features["brand_in_subdomain"] = 0
    features["brand_in_path"] = 0
    features["suspecious_tld"] = 0
    features["statistical_report"] = 0

    return features


# ---------------- CONTENT FEATURE EXTRACTION ----------------
def extract_content_features(url: str) -> dict:
    features = {
        "nb_hyperlinks": 0,
        "ratio_intHyperlinks": 0,
        "ratio_extHyperlinks": 0,
        "ratio_nullHyperlinks": 0,
        "nb_extCSS": 0,
        "ratio_intRedirection": 0,
        "ratio_extRedirection": 0,
        "ratio_intErrors": 0,
        "ratio_extErrors": 0,
        "login_form": 0,
        "external_favicon": 0,
        "links_in_tags": 0,
        "submit_email": 0,
        "ratio_intMedia": 0,
        "ratio_extMedia": 0,
        "sfh": 0,
        "iframe": 0,
        "popup_window": 0,
        "safe_anchor": 0,
        "onmouseover": 0,
        "right_clic": 0,
        "empty_title": 0,
        "domain_in_title": 0,
        "domain_with_copyright": 0,
    }

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return features

        response = requests.get(url, timeout=8, headers=headers)
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        base_domain = get_domain(url)

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        features["empty_title"] = 1 if not title else 0
        features["domain_in_title"] = 1 if base_domain and base_domain.split(".")[0].lower() in title.lower() else 0

        forms = soup.find_all("form")
        features["login_form"] = 1 if forms else 0

        for form in forms:
            action = (form.get("action") or "").strip().lower()
            if action == "" or action == "about:blank":
                features["sfh"] = 1
            if "mailto:" in action:
                features["submit_email"] = 1

        features["iframe"] = 1 if soup.find_all("iframe") else 0

        if "window.open(" in html:
            features["popup_window"] = 1
        if "onmouseover" in html:
            features["onmouseover"] = 1
        if "event.button==2" in html or "contextmenu" in html:
            features["right_clic"] = 1

        favicons = soup.find_all("link", rel=lambda x: x and "icon" in str(x).lower())
        for icon in favicons:
            href = icon.get("href", "")
            if href.startswith("http"):
                icon_domain = get_domain(href)
                if icon_domain and icon_domain != base_domain:
                    features["external_favicon"] = 1

        links = soup.find_all("a", href=True)
        total_links = len(links)
        features["nb_hyperlinks"] = total_links

        internal_links = 0
        external_links = 0
        null_links = 0
        safe_anchor_count = 0

        for link in links:
            href = link.get("href", "").strip().lower()

            if href in ["", "#", "#content", "javascript:void(0)"]:
                null_links += 1
            elif href.startswith("http"):
                link_domain = get_domain(href)
                if link_domain == base_domain:
                    internal_links += 1
                else:
                    external_links += 1
            else:
                internal_links += 1

            if href not in ["#", "javascript:void(0)", ""]:
                safe_anchor_count += 1

        features["ratio_intHyperlinks"] = safe_div(internal_links, total_links)
        features["ratio_extHyperlinks"] = safe_div(external_links, total_links)
        features["ratio_nullHyperlinks"] = safe_div(null_links, total_links)
        features["safe_anchor"] = safe_div(safe_anchor_count, total_links)

        css_links = soup.find_all("link", rel=lambda x: x and "stylesheet" in str(x).lower(), href=True)
        ext_css = 0
        for css in css_links:
            href = css.get("href", "")
            if href.startswith("http"):
                css_domain = get_domain(href)
                if css_domain != base_domain:
                    ext_css += 1
        features["nb_extCSS"] = ext_css

        media_tags = soup.find_all(["img", "audio", "embed", "iframe", "video", "source"])
        int_media = 0
        ext_media = 0
        total_media = 0

        for tag in media_tags:
            src = tag.get("src", "")
            if not src:
                continue
            total_media += 1
            if src.startswith("http"):
                media_domain = get_domain(src)
                if media_domain == base_domain:
                    int_media += 1
                else:
                    ext_media += 1
            else:
                int_media += 1

        features["ratio_intMedia"] = safe_div(int_media, total_media)
        features["ratio_extMedia"] = safe_div(ext_media, total_media)

        tags_with_links = soup.find_all(["meta", "script", "link"])
        tag_links = 0
        ext_tag_links = 0
        for tag in tags_with_links:
            val = tag.get("src") or tag.get("href")
            if not val:
                continue
            tag_links += 1
            if val.startswith("http"):
                link_domain = get_domain(val)
                if link_domain != base_domain:
                    ext_tag_links += 1
        features["links_in_tags"] = safe_div(ext_tag_links, tag_links)

        body_text = soup.get_text(" ", strip=True).lower()
        if "copyright" in body_text or "©" in body_text:
            features["domain_with_copyright"] = 1 if base_domain and base_domain.split(".")[0].lower() in body_text else 0

    except requests.RequestException:
        pass
    except Exception:
        pass

    return features


# ---------------- ALL FEATURES ----------------
def extract_all_features_local(url: str) -> dict:
    features = {}
    features.update(extract_url_features(url))
    features.update(extract_content_features(url))

    features.setdefault("whois_registered_domain", 0)
    features.setdefault("domain_registration_length", 0)
    features.setdefault("domain_age", 0)
    features.setdefault("web_traffic", 0)
    features.setdefault("dns_record", 0)
    features.setdefault("google_index", 0)
    features.setdefault("page_rank", 0)

    return features


# ---------------- RESULT HELPERS ----------------
def get_top_feature_values(input_df, top_n=8):
    row = input_df.iloc[0].to_dict()
    non_zero_items = [(k, v) for k, v in row.items() if v not in [0, 0.0, "", None]]

    non_zero_items = sorted(
        non_zero_items,
        key=lambda x: float(x[1]) if isinstance(x[1], (int, float, np.integer, np.floating)) else 0,
        reverse=True
    )

    return non_zero_items[:top_n]


def get_individual_model_probs(input_df):
    rf_prob = 0.0
    xgb_prob = 0.0
    mlp_prob = 0.0

    try:
        estimators = dict(model.named_estimators_)

        if "rf" in estimators:
            rf_prob = float(estimators["rf"].predict_proba(input_df)[0][1])

        if "xgb" in estimators:
            xgb_prob = float(estimators["xgb"].predict_proba(input_df)[0][1])

        if "mlp" in estimators:
            mlp_prob = float(estimators["mlp"].predict_proba(input_df)[0][1])

    except Exception as e:
        print("Error getting individual model probabilities:", e)

    return rf_prob, xgb_prob, mlp_prob


def save_scan_history(user_id, url, prediction, probability, risk):
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO scan_history (user_id, url, prediction, probability, risk)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, url, prediction, probability, risk))
    conn.commit()
    conn.close()


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/contact", methods=["POST"])
def contact():
    name = request.form["name"]
    email = request.form["email"].strip().lower()
    subject = request.form["subject"]
    message = request.form["message"]

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO contact_messages (name, email, subject, message)
        VALUES (?, ?, ?, ?)
    """, (name, email, subject, message))

    conn.commit()
    conn.close()

    flash("Your message has been sent successfully.", "success")
    return redirect(url_for("home"))


#---------------Admin Messages--------------#
@app.route("/admin/messages")
def admin_messages():
    if "user_id" not in session:
        flash("Please login first.", "danger")
        return redirect(url_for("login"))

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    cur.execute("""
        SELECT id, name, email, subject, message, created_at
        FROM contact_messages
        ORDER BY id DESC
    """)
    messages = cur.fetchall()
    conn.close()

    return render_template("admin_messages.html", messages=messages)


# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        if len(password) < 8:
            flash("Password must be at least 8 characters long.", "danger")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        existing_user = cur.fetchone()

        if existing_user:
            conn.close()
            flash("Email already registered.", "danger")
            return redirect(url_for("register"))

        cur.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                    (name, email, hashed_password))
        conn.commit()
        conn.close()

        flash("Registration successful!", "success")
        return redirect(url_for("register"))

    return render_template("register.html")


# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session.clear()
            session["user_id"] = user[0]
            session["user_name"] = user[1]
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


# ---------------- DASHBOARD ----------------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM scan_history WHERE user_id = ?", (user_id,))
    total_urls = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM scan_history WHERE user_id = ? AND risk = ?", (user_id, "High Risk"))
    high_risk = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM scan_history WHERE user_id = ? AND prediction = ?", (user_id, "Legitimate"))
    safe = cur.fetchone()[0]

    conn.close()

    if request.method == "POST":
        url = request.form["url"].strip()

        if len(url) > 500:
            flash("URL is too long.", "danger")
            return render_template(
                "dashboard.html",
                user_name=session.get("user_name"),
                total_urls=total_urls,
                high_risk=high_risk,
                safe=safe
            )

        if model is None:
            flash("Model not loaded. Please check model file path.", "danger")
            return render_template(
                "dashboard.html",
                user_name=session.get("user_name"),
                total_urls=total_urls,
                high_risk=high_risk,
                safe=safe
            )

        try:
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "http://" + url

            if not is_valid_url(url):
                flash("Please enter a valid URL.", "danger")
                return render_template(
                    "dashboard.html",
                    user_name=session.get("user_name"),
                    total_urls=total_urls,
                    high_risk=high_risk,
                    safe=safe
                )

            # Use imported extractor if available, else fallback local extractor
            try:
                input_features = extract_all_features(url)
            except Exception:
                input_features = extract_all_features_local(url)

            input_df = pd.DataFrame([input_features])

            if feature_names is not None:
                input_df = input_df.reindex(columns=feature_names, fill_value=0)

            prediction = model.predict(input_df)[0]
            final_prob = float(model.predict_proba(input_df)[0][1])

            rf_prob, xgb_prob, mlp_prob = get_individual_model_probs(input_df)
            risk = get_risk_level(final_prob)
            label = "Phishing" if prediction == 1 else "Legitimate"
            top_features = get_top_feature_values(input_df, top_n=8)

            result = {
                "url": url,
                "prediction": label,
                "risk": risk,
                "probability": round(final_prob * 100, 2),
                "rf_prob": round(rf_prob * 100, 2),
                "xgb_prob": round(xgb_prob * 100, 2),
                "mlp_prob": round(mlp_prob * 100, 2),
                "top_features": top_features
            }

            session["last_result"] = result

            save_scan_history(
                user_id,
                url,
                label,
                round(final_prob * 100, 2),
                risk
            )

            return redirect(url_for("result_page"))

        except Exception as e:
            print("Prediction error:", e)
            flash("An error occurred while analyzing the URL.", "danger")
            return render_template(
                "dashboard.html",
                user_name=session.get("user_name"),
                total_urls=total_urls,
                high_risk=high_risk,
                safe=safe
            )

    return render_template(
        "dashboard.html",
        user_name=session.get("user_name"),
        total_urls=total_urls,
        high_risk=high_risk,
        safe=safe
    )


#-------Result page------
@app.route("/result")
def result_page():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if "last_result" not in session:
        flash("No analysis result found. Please analyze a URL first.", "danger")
        return redirect(url_for("dashboard"))

    return render_template(
        "result.html",
        result=session["last_result"],
        user_name=session.get("user_name")
    )


#---------Function-------
def convert_html_to_pdf(source_html):
    result = BytesIO()
    pdf = pisa.CreatePDF(source_html, dest=result)

    if not pdf.err:
        return result.getvalue()
    return None


# -------- REPORT PAGE --------
@app.route("/report")
def report():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if "last_result" not in session:
        flash("No report data found.", "danger")
        return redirect(url_for("dashboard"))

    result = session["last_result"]
    return render_template("report.html", result=result)


# -------- DOWNLOAD REPORT PDF --------
@app.route("/download_report")
def download_report():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if "last_result" not in session:
        flash("No report available.", "danger")
        return redirect(url_for("dashboard"))

    result = session["last_result"]

    html = render_template("report_pdf.html", result=result)
    pdf = convert_html_to_pdf(html)

    if pdf:
        response = make_response(pdf)
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = "attachment; filename=phishing_report.pdf"
        return response
    else:
        flash("Error generating PDF.", "danger")
        return redirect(url_for("dashboard"))


# ---------------- HISTORY PAGE ----------------
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT url, prediction, probability, risk, scanned_at
        FROM scan_history
        WHERE user_id = ?
        ORDER BY id DESC
    """, (session["user_id"],))
    history_data = cur.fetchall()
    conn.close()

    return render_template("history.html", history=history_data, user_name=session["user_name"])


# ---------------- LOGOUT --------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)