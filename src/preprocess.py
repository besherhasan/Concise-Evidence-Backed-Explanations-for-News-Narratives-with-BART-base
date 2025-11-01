import pandas as pd
import re
import torch
from sentence_transformers import SentenceTransformer, util

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_contextual_sentences(article, narrative, subnarrative, max_sentences=3):
    sentences = re.split(r'(?<=[.!?]) +', article)  # Split article into sentences
    query = f"{narrative} {subnarrative}"
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    sentence_embeddings = embed_model.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, sentence_embeddings).squeeze(0)
    top_indices = torch.topk(scores, k=min(max_sentences, len(sentences))).indices
    return " ".join([sentences[i] for i in top_indices])

def preprocess_data(df):
    inputs, targets = [], []
    for _, row in df.iterrows():
        dominant_narrative = row['dominant_narrative']
        dominant_subnarrative = row['dominant_subnarrative'] if pd.notna(row['dominant_subnarrative']) else "None"
        article_text = extract_contextual_sentences(
            row['article_text'], dominant_narrative, dominant_subnarrative
        )
        explanation = row['explanation']

        input_text = (
            f"<NARRATIVE> {dominant_narrative} </NARRATIVE> "
            f"<SUB-NARRATIVE> {dominant_subnarrative} </SUB-NARRATIVE> "
            f"<ARTICLE> {article_text} </ARTICLE>"
        )
        inputs.append(input_text)
        targets.append(explanation)

    return inputs, targets

if __name__ == "__main__":
    df = pd.read_excel("../data/combined_train_set_with_splits.xlsx")
    train_data = df[df['split'] == 'train']
    val_data = df[df['split'] == 'validation']

    train_inputs, train_targets = preprocess_data(train_data)
    val_inputs, val_targets = preprocess_data(val_data)

    # Save preprocessed data
    pd.DataFrame({'inputs': train_inputs, 'targets': train_targets}).to_csv("../data/train_preprocessed.csv", index=False)
    pd.DataFrame({'inputs': val_inputs, 'targets': val_targets}).to_csv("../data/val_preprocessed.csv", index=False)
