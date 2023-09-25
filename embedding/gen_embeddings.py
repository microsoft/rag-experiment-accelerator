
from sentence_transformers import SentenceTransformer



def generate_embedding(size, chunk):
    if size == 384:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    elif size == 768:
        model = SentenceTransformer('all-mpnet-base-v2')
    elif size == 1024:
        model = SentenceTransformer('bert-large-nli-mean-tokens')
    else:
        model = ""
    return model.encode([str(chunk)]).tolist()
