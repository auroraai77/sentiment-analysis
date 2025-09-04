import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
import pandas as pd

def evaluate(path="sample_reviews.csv"):
    df = pd.read_csv(path)
    texts, labels = df["review"].tolist(), df["label"].tolist()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("./models/best_model")

    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

    outputs = model(**encodings)
    preds = torch.argmax(outputs.logits, dim=-1).tolist()

    print(classification_report(labels, preds))

if __name__ == "__main__":
    evaluate()
