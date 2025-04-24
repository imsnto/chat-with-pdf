from langchain_chroma import Chroma
from langchain_core.documents import Document


def to_document(chunk):
    return Document(page_content=chunk)


def get_vector_store(chunks, embeddings):
    # Convert chunks to documents first
    documents = [to_document(chunk) for chunk in chunks]

    # Create vector store and add documents
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory='./chroma'
    )

    # Generate IDs as strings from 0 to len(documents)-1
    document_ids = [str(idx) for idx in range(len(documents))]

    # Add documents to the vector store
    vector_store.add_documents(documents=documents, ids=document_ids)

    return vector_store
