import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

def load_data(path="sample_reviews.csv"):
    df = pd.read_csv(path)
    return df["review"].tolist(), df["label"].tolist()

def train():
    reviews, labels = load_data()
    tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
    encodings = tokenizer(reviews, truncation=True, padding=True)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(encodings["input_ids"]),
        torch.tensor(encodings["attention_mask"]),
        torch.tensor(labels)
    )

    model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_total_limit=1,
        logging_steps=5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

if __name__ == "__main__":
    train()

