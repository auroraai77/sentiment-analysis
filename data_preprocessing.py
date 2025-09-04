import re
import spacy
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = [tok.text for tok in nlp(text) if tok.text not in stop_words]
    return " ".join(tokens)
