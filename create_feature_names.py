import joblib
from feature_extractor import extract_all_features

# Dummy URL
url = "https://example.com"

features = extract_all_features(url)

feature_names = list(features.keys())

joblib.dump(feature_names, "model/feature_names.pkl")

print("✅ feature_names.pkl created successfully")
print("Total features:", len(feature_names))