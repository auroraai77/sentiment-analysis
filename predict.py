import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class SentimentAnalyzer:
    def __init__(self):
        model_path = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        return "Positive" if pred == 1 else "Negative"

if __name__ == "__main__":
    sa = SentimentAnalyzer()
    print(sa.predict("I really love this product!"))
    print(sa.predict("This is the worst thing I ever bought."))
