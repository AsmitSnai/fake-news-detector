import streamlit as st
import pickle
from utils import clean_text

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector")
st.markdown("Paste a news article below to check whether it is **Real** or **Fake**.")

user_input = st.text_area("Enter News Text Here", height=200)

if st.button("Check News"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        vector_input = vectorizer.transform([cleaned_text])

        prediction = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0]

        if prediction == 1:
            st.success("Prediction: REAL NEWS ✅")
            st.write(f"Confidence: {round(probability[1]*100,2)}%")
        else:
            st.error("Prediction: FAKE NEWS ❌")
            st.write(f"Confidence: {round(probability[0]*100,2)}%")