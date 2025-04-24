from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


def to_document(chunk):
    return Document(page_content=chunk)


def get_vector_store(chunks, embeddings):
    # Convert chunks to documents first
    documents = [to_document(chunk) for chunk in chunks]


    # # Configure persistent Chroma with a valid persist_directory
    # persist_directory = "./chroma_db"
    # client_settings = chromadb.config.Settings(
    #     #persist_directory=persist_directory,
    #     is_persistent=False
    # )
    #
    # # Initialize Chroma vector store
    # vector_store = Chroma(
    #     embedding_function=embeddings,
    #     client_settings=client_settings,
    #     collection_name="my_collection"
    # )

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Generate IDs as strings from 0 to len(documents)-1
    document_ids = [str(idx) for idx in range(len(documents))]

    # Add documents to the vector store
    vector_store.add_documents(documents=documents, ids=document_ids)

    return vector_store
