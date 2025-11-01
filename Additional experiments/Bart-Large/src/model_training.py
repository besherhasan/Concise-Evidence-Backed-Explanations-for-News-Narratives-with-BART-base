import torch
from tqdm import tqdm
from transformers import get_scheduler

def train_model(model, train_loader, optimizer, scheduler, num_epochs, device):
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
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

def validate_model(model, val_loader, tokenizer, device):
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=80)
            decoded_preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
            decoded_refs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]

            predictions.extend(decoded_preds)
            references.extend(decoded_refs)
    return predictions, references
