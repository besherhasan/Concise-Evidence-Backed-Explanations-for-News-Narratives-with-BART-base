import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, get_scheduler
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from dataset import NarrativeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    train_data = pd.read_csv("../data/train_preprocessed.csv")
    val_data = pd.read_csv("../data/val_preprocessed.csv")

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)

    train_dataset = NarrativeDataset(train_data['inputs'], train_data['targets'], tokenizer)
    val_dataset = NarrativeDataset(val_data['inputs'], val_data['targets'], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 10
    num_training_steps = num_epochs * len(train_loader)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    model.save_pretrained("../models/best_bart_model_assignment2")
    tokenizer.save_pretrained("../models/best_bart_model_assignment2")
    print("Model and tokenizer saved.")

if __name__ == "__main__":
    train()
