import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from bert_score import score
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    val_data = pd.read_csv("../data/val_preprocessed.csv")
    tokenizer = BartTokenizer.from_pretrained("../models/best_bart_model_assignment2")
    model = BartForConditionalGeneration.from_pretrained("../models/best_bart_model_assignment2").to(device)

    predictions, references = [], []
    for _, row in tqdm(val_data.iterrows(), total=len(val_data), desc="Evaluating"):
        inputs = tokenizer(row['inputs'], return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=80
        )
        predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        references.append(row['targets'])

    P, R, F1 = score(predictions, references, lang="en", verbose=True)
    print(f"BERTScore Results:\nPrecision: {P.mean():.4f}\nRecall: {R.mean():.4f}\nF1: {F1.mean():.4f}")

if __name__ == "__main__":
    evaluate()
