from flask import Flask, render_template, request, jsonify
import joblib
import os
import urllib.request
import zipfile

app = Flask(__name__)

# -----------------------------
# Download & load model
# -----------------------------
MODEL_ZIP_URL = "https://drive.google.com/uc?id=1hu7pzjNYaQewda2mKLM29L4qeS19g8DB"
MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    print("⬇️ Downloading model from Drive...")
    urllib.request.urlretrieve(MODEL_ZIP_URL, "models.zip")

    with zipfile.ZipFile("models.zip", "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    os.remove("models.zip")
    print("✅ Model downloaded and extracted")

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# -----------------------------
# Rule-based check
# -----------------------------
def rule_based_check(text):
    rules = [
        "is dead", "dies", "killed", "death of",
        "miracle cure", "you won", "lottery",
        "fraud", "scam"
    ]
    text = text.lower()
    return any(rule in text for rule in rules)

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify():
    text = request.form.get("news", "").strip()

    if not text:
        return jsonify({"result": "Please enter text", "confidence": 0})

    if rule_based_check(text):
        return jsonify({"result": "FAKE NEWS 🔴", "confidence": 95})

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = round(max(proba) * 100, 2)

    result = "REAL NEWS 🟢" if pred == 1 else "FAKE NEWS 🔴"
    return jsonify({"result": result, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)