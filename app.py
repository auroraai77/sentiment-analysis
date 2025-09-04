import streamlit as st
from predict import SentimentAnalyzer

st.title("📝 Sentiment Analysis of Product Reviews")
st.write("Classify reviews as Positive, Neutral, or Negative")

analyzer = SentimentAnalyzer()

user_input = st.text_area("Enter a review:")
if st.button("Analyze"):
    if user_input.strip():
        prediction = analyzer.predict(user_input)
        st.success(f"Prediction: **{prediction}**")
    else:
        st.warning("Please enter some text!")
