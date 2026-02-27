import pickle
from utils import clean_text

# Load saved model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("===================================")
print("     FAKE NEWS DETECTOR READY      ")
print("===================================")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter news text: ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    # Clean text (VERY IMPORTANT)
    cleaned_text = clean_text(user_input)

    # Convert to vector
    vector_input = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(vector_input)[0]
    probability = model.predict_proba(vector_input)[0]

    if prediction == 1:
        print(f"Prediction: REAL NEWS ✅")
        print(f"Confidence: {round(probability[1] * 100, 2)}%")
    else:
        print(f"Prediction: FAKE NEWS ❌")
        print(f"Confidence: {round(probability[0] * 100, 2)}%")

    print("-" * 40)