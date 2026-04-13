import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import pyarrow.parquet as pq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

data = pq.read_table('../data/data_activelearning.parquet')
data = data.to_pandas()

schizophrenia_articles = data.query('id_subcorpus == "psyB"')["text"].tolist()
depression_articles = data.query('id_subcorpus == "psyC"')["text"].tolist()

# Encode schizophrenia articles
schizophrenia_embeddings = model.encode(schizophrenia_articles, convert_to_tensor=False)
depression_embeddings = model.encode(depression_articles, convert_to_tensor=False)

print(f"Shape of embeddings: {schizophrenia_embeddings.shape}")
print(f"Embedding dimension: {schizophrenia_embeddings.shape[1]}")

print(f"Shape of depression embeddings: {depression_embeddings.shape}")
print(f"Depression embedding dimension: {depression_embeddings.shape[1]}")


np.savez("../data/schizophrenia_embeddings.npz", schizophrenia_embeddings)
np.savez("../data/depression_embeddings.npz", depression_embeddings)