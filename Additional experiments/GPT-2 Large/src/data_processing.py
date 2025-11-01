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
        f"<TASK> Generate Explanation\n"
        f"<NARRATIVE> {dominant_narrative}\n"
        f"<SUB-NARRATIVE> {dominant_subnarrative}\n"
        f"<ARTICLE>\n{article_text}\n"
        f"<EXPLANATION>"
    )
    target_text = explanation
    return input_text, target_text

def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)
    train_data = df[df['split'] == 'train']
    val_data = df[df['split'] == 'validation']

    train_inputs, train_targets = zip(*train_data.apply(preprocess_data, axis=1))
    val_inputs, val_targets = zip(*val_data.apply(preprocess_data, axis=1))

    return train_inputs, train_targets, val_inputs, val_targets
