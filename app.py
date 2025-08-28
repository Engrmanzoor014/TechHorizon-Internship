# app.py
import re
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download resources (only runs the first time)
nltk.download("stopwords")
nltk.download("wordnet")

# prepare tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# preprocessing function
def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)  # keep only letters
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)


# ---------------- Streamlit UI ---------------- #
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter news text:")

if st.button("Predict"):
    processed = preprocess(user_input)
    vectorized = vectorizer.transform([processed])
    prediction = models["Logistic Regression"].predict(vectorized)[0]
    st.write("✅ Real News" if prediction == 1 else "❌ Fake News")
