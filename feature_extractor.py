import re
import requests
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import tldextract


def safe_div(a, b):
    return a / b if b != 0 else 0


def get_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


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

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, timeout=8, headers=headers)
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        base_domain = get_domain(url)

        # Title features
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        features["empty_title"] = 1 if not title else 0
        features["domain_in_title"] = 1 if base_domain and base_domain.split(".")[0].lower() in title.lower() else 0

        # Forms
        forms = soup.find_all("form")
        features["login_form"] = 1 if forms else 0

        for form in forms:
            action = (form.get("action") or "").strip().lower()
            if action == "" or action == "about:blank":
                features["sfh"] = 1
            if "mailto:" in action:
                features["submit_email"] = 1

        # iframe
        features["iframe"] = 1 if soup.find_all("iframe") else 0

        # popup / JS signals
        if "window.open(" in html:
            features["popup_window"] = 1
        if "onmouseover" in html:
            features["onmouseover"] = 1
        if "event.button==2" in html or "contextmenu" in html:
            features["right_clic"] = 1

        # Favicon
        favicons = soup.find_all("link", rel=lambda x: x and "icon" in str(x).lower())
        for icon in favicons:
            href = icon.get("href", "")
            if href.startswith("http"):
                icon_domain = get_domain(href)
                if icon_domain and icon_domain != base_domain:
                    features["external_favicon"] = 1

        # Links
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

        # CSS
        css_links = soup.find_all("link", rel=lambda x: x and "stylesheet" in str(x).lower(), href=True)
        ext_css = 0
        for css in css_links:
            href = css.get("href", "")
            if href.startswith("http"):
                css_domain = get_domain(href)
                if css_domain != base_domain:
                    ext_css += 1
        features["nb_extCSS"] = ext_css

        # Media
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

        # links in tags
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

        # copyright/domain
        body_text = soup.get_text(" ", strip=True).lower()
        if "copyright" in body_text or "©" in body_text:
            features["domain_with_copyright"] = 1 if base_domain.split(".")[0].lower() in body_text else 0

    except requests.RequestException:
        pass
    except Exception:
        pass

    return features


def extract_all_features(url: str) -> dict:
    features = {}
    features.update(extract_url_features(url))
    features.update(extract_content_features(url))

    # placeholders for features you are not yet extracting live
    features.setdefault("whois_registered_domain", 0)
    features.setdefault("domain_registration_length", 0)
    features.setdefault("domain_age", 0)
    features.setdefault("web_traffic", 0)
    features.setdefault("dns_record", 0)
    features.setdefault("google_index", 0)
    features.setdefault("page_rank", 0)

    return features