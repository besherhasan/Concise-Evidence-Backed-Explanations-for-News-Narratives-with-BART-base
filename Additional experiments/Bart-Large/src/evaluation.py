from bert_score import score

def evaluate(predictions, references):
    P, R, F1 = score(predictions, references, lang="en", verbose=True)
    print(f"BERTScore Results:\nPrecision: {P.mean():.4f}\nRecall: {R.mean():.4f}\nF1: {F1.mean():.4f}")
