import pandas as pd
import re

def load_data(filepath):
    return pd.read_excel(filepath)

def extract_contextual_sentences(article, narrative, subnarrative, embed_model, max_sentences=3):
    sentences = re.split(r'(?<=[.!?]) +', article)
    query = f"{narrative} {subnarrative}"
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    sentence_embeddings = embed_model.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, sentence_embeddings).squeeze(0)
    top_indices = torch.topk(scores, k=min(max_sentences, len(sentences))).indices
    return " ".join([sentences[i] for i in top_indices])

def preprocess_data(row, embed_model):
    dominant_narrative = row['dominant_narrative']
    dominant_subnarrative = row['dominant_subnarrative'] if pd.notna(row['dominant_subnarrative']) else "None"
    article_text = extract_contextual_sentences(
        row['article_text'], dominant_narrative, dominant_subnarrative, embed_model
    )
    explanation = row['explanation']

    input_text = (
        f"<TASK> Generate Explanation\n"
        f"<NARRATIVE> {dominant_narrative}\n"
        f"<SUB-NARRATIVE> {dominant_subnarrative}\n"
        f"<ARTICLE>\n{article_text}\n"
        f"<EXPLANATION>"
    )
    target_text = explanation
    return input_text, target_text
