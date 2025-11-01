import torch
from tqdm import tqdm
from transformers import get_scheduler

def train_model(model, train_loader, optimizer, scheduler, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

def validate_model(model, val_loader, tokenizer, device):
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=80, num_beams=5, length_penalty=1.2, early_stopping=True
            )
            predictions.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs])
            references.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in labels])
    return predictions, references
