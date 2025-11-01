import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate():
    new_data = pd.read_excel("../data/combined_dev_set.xlsx")
    tokenizer = BartTokenizer.from_pretrained("../models/best_bart_model_assignment2")
    model = BartForConditionalGeneration.from_pretrained("../models/best_bart_model_assignment2").to(device)

    def generate_explanation(row):
        dominant_narrative = row['dominant_narrative']
        dominant_subnarrative = row['dominant_subnarrative'] if pd.notna(row['dominant_subnarrative']) else "None"
        article_text = row['article_text']

        input_text = (
            f"<NARRATIVE> {dominant_narrative} </NARRATIVE> "
            f"<SUB-NARRATIVE> {dominant_subnarrative} </SUB-NARRATIVE> "
            f"<ARTICLE> {article_text} </ARTICLE>"
        )

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=80,
            num_beams=4, no_repeat_ngram_size=3, early_stopping=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    tqdm.pandas(desc="Generating Explanations")
    new_data['generated_explanation'] = new_data.progress_apply(generate_explanation, axis=1)
    new_data.to_excel("../data/combined_dev_set_with_explanations.xlsx", index=False)

if __name__ == "__main__":
    generate()
