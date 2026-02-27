import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils import clean_text

# Load dataset
fake = pd.read_csv("data/fake.csv")
real = pd.read_csv("data/true.csv")

# Add labels
fake["label"] = 0   # Fake
real["label"] = 1   # Real

# Combine & shuffle
data = pd.concat([fake, real], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset Loaded Successfully!")
print("Label Distribution:")
print(data["label"].value_counts())

# Clean text column
data["clean_text"] = (data["title"] + " " + data["text"]).apply(clean_text)

# Feature extraction (Improved)
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model (Balanced)
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel trained and saved successfully!")