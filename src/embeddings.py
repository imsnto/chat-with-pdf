from langchain_huggingface import HuggingFaceEmbeddings
import torch

def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model_name = "sentence-transformers/all-mpnet-base-v2"
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

