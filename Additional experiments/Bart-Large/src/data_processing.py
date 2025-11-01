import pandas as pd
from scripts.contextual_extraction import extract_contextual_sentences

def preprocess_data(row):
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
    return input_text, explanation

def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)
    train_data = df[df['split'] == 'train']
    val_data = df[df['split'] == 'validation']

    train_inputs, train_targets = zip(*train_data.apply(preprocess_data, axis=1))
    val_inputs, val_targets = zip(*val_data.apply(preprocess_data, axis=1))

    return train_inputs, train_targets, val_inputs, val_targets
