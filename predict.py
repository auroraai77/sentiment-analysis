import torch
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentAnalyzer:
    def __init__(self, model_path="./models/best_model"):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained(model_path)

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        return ["Negative", "Neutral", "Positive"][pred]

if __name__ == "__main__":
    sa = SentimentAnalyzer()
    print(sa.predict("I really love this product!"))
