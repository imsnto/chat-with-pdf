# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings

def get_embeddings():
    #model_name = "sentence-transformers/all-mpnet-base-v2"
    model_name = "embed-english-v3.0"
    return CohereEmbeddings(model=model_name)

