import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

def load_data(path="sample_reviews.csv"):
    df = pd.read_csv(path)
    return df["review"].tolist(), df["label"].tolist()

def train():
    reviews, labels = load_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    encodings = tokenizer(reviews, truncation=True, padding=True)
    labels = torch.tensor(labels)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(encodings["input_ids"]),
        torch.tensor(encodings["attention_mask"]),
        labels
    )

    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    training_args = TrainingArguments(
        output_dir="./models",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    model.save_pretrained("./models/best_model")

if __name__ == "__main__":
    train()
