from transformers import BartTokenizer, BartForConditionalGeneration

def load_bart_model(model_name="facebook/bart-large", device="cpu"):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    return tokenizer, model
