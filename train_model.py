import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Labels
fake["label"] = 0
real["label"] = 1

# Combine + shuffle
data = pd.concat([fake, real])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Use title + text
X = data["title"].astype(str) + " " + data["text"].astype(str)
y = data["label"]

# 🔥 TF-IDF with FEATURE LIMIT (MOST IMPORTANT)
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,      # 🔥 SIZE CONTROL
    max_df=0.7,
    ngram_range=(1, 1)
)

X_vec = vectorizer.fit_transform(X)

# Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)
model.fit(X_vec, y)

# Save model & vectorizer
joblib.dump(model, "models/fake_news_model.pkl", compress=3)
joblib.dump(vectorizer, "models/vectorizer.pkl", compress=3)

print("✅ Model trained & saved successfully")
