import streamlit as st
from predict import SentimentAnalyzer

st.set_page_config(page_title="Sentiment Analysis App", page_icon="📝", layout="centered")

st.title("📝 Sentiment Analysis of Product Reviews")
st.markdown(
    """
Analyze customer feedback and classify it as:
- ✅ **Positive**
- ❌ **Negative**
    """
)

@st.cache_resource
def load_analyzer():
    return SentimentAnalyzer()

analyzer = load_analyzer()

st.subheader("🔍 Single Review")
user_input = st.text_area("Enter a product review:")
if st.button("Analyze Single"):
    if user_input.strip():
        prediction = analyzer.predict(user_input)
        if prediction == "Positive":
            st.success(f"✅ Prediction: **{prediction}**")
        else:
            st.error(f"❌ Prediction: **{prediction}**")
    else:
        st.warning("⚠️ Please enter some text!")

st.subheader("📊 Batch Analysis")
multi_input = st.text_area("Paste multiple reviews (one per line):")
if st.button("Analyze Batch"):
    if multi_input.strip():
        reviews = [r.strip() for r in multi_input.strip().split("\n") if r.strip()]
        results = [analyzer.predict(r) for r in reviews]
        pos_count = results.count("Positive")
        neg_count = results.count("Negative")

        st.write("### Results:")
        for r, pred in zip(reviews, results):
            icon = "✅" if pred == "Positive" else "❌"
            st.write(f"- {r} → {icon} **{pred}**")

        st.info(f"**Summary:** {pos_count} Positive ✅ | {neg_count} Negative ❌")
    else:
        st.warning("⚠️ Please enter at least one review!")

with st.expander("ℹ️ About"):
    st.write("This app uses a DistilBERT model fine-tuned on SST-2 to classify sentiment of product reviews.")

