import re
from sentence_transformers import SentenceTransformer, util
import torch

# Load pre-trained sentence embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_contextual_sentences(article, narrative, subnarrative, max_sentences=3):
    sentences = re.split(r'(?<=[.!?]) +', article)
    query = f"{narrative} {subnarrative}"
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    sentence_embeddings = embed_model.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, sentence_embeddings).squeeze(0)
    top_indices = torch.topk(scores, k=min(max_sentences, len(sentences))).indices
    return " ".join([sentences[i] for i in top_indices])
