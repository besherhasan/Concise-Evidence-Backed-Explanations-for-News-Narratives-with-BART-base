import pandas as pd
from torch.utils.data import DataLoader
from src.data_processing import load_data, preprocess_data
from src.embedding import get_embedding_model
from src.dataset import NarrativeDataset
from src.model import get_model_and_tokenizer, get_optimizer_and_scheduler
from src.evaluation import validate_model, compute_bert_score

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = get_embedding_model()

df = load_data("data/combined_train_set_with_splits.xlsx")
train_data = df[df['split'] == 'train']
val_data = df[df['split'] == 'validation']
train_inputs, train_targets = zip(*train_data.apply(preprocess_data, axis=1, embed_model=embed_model))
val_inputs, val_targets = zip(*val_data.apply(preprocess_data, axis=1, embed_model=embed_model))

model, tokenizer = get_model_and_tokenizer(device=device)

train_dataset = NarrativeDataset(train_inputs, train_targets, tokenizer)
val_dataset = NarrativeDataset(val_inputs, val_targets, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

optimizer, scheduler = get_optimizer_and_scheduler(model, train_loader, num_epochs=10)

for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/10"):
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
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# Evaluate the model
predictions, references = validate_model(model, val_loader, tokenizer, device)
compute_bert_score(predictions, references)
