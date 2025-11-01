from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_gpt_model(model_name="gpt2-large", device="cpu"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    # Add a pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return tokenizer, model
