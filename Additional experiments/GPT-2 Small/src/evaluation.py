from bert_score import score
from tqdm import tqdm

def validate_model(model, val_loader, tokenizer, device):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=80,
                num_beams=5,
                length_penalty=1.2,
                early_stopping=True
            )
            decoded_preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
            decoded_refs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]

            predictions.extend(decoded_preds)
            references.extend(decoded_refs)

    return predictions, references

def compute_bert_score(predictions, references, lang="en"):
    P, R, F1 = score(predictions, references, lang=lang, verbose=True)
    print(f"BERTScore Results:\nPrecision: {P.mean():.4f}\nRecall: {R.mean():.4f}\nF1: {F1.mean():.4f}")
