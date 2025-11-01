import torch
from torch.utils.data import DataLoader
from scripts.data_processing import load_and_preprocess_data
from scripts.dataset import NarrativeDataset
from scripts.model_training import train_model, validate_model
from scripts.evaluation import evaluate
from models.gpt_model import load_gpt_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "data/combined_train_set_with_splits.xlsx"
batch_size = 8
num_epochs = 3


train_inputs, train_targets, val_inputs, val_targets = load_and_preprocess_data(data_path)


tokenizer, model = load_gpt_model(model_name="gpt2-medium", device=device)



train_dataset = NarrativeDataset(train_inputs, train_targets, tokenizer)
val_dataset = NarrativeDataset(val_inputs, val_targets, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_loader)
scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)


train_model(model, train_loader, optimizer, scheduler, num_epochs, device)
predictions, references = validate_model(model, val_loader, tokenizer, device)


evaluate(predictions, references)
