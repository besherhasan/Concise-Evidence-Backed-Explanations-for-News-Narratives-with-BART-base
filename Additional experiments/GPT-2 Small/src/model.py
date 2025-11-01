import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_scheduler
from torch.optim import AdamW

def get_model_and_tokenizer(model_name="gpt2", device="cpu"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer

def get_optimizer_and_scheduler(model, train_loader, num_epochs, lr=5e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )
    return optimizer, scheduler
