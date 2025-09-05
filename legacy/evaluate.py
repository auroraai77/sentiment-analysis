import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report

def evaluate(path="sample_reviews.csv"):
    df = pd.read_csv(path)
    texts, labels = df["review"].tolist(), df["label"].tolist()

    tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**encodings)
    preds = torch.argmax(outputs.logits, dim=-1).tolist()

    print(classification_report(labels, preds))

if __name__ == "__main__":
    evaluate()

