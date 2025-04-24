from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Constants
DEFAULT_MODEL = "llama-3.1-8b-instant"
SYSTEM_TEMPLATE = "You are a helpful assistant. You will answer questions based on the following context:\n\n{context}"

def create_qa_prompt() -> ChatPromptTemplate:
    """Create the question-answering prompt template."""
    return ChatPromptTemplate(
        input_variables=["context", "input"],
        messages=[
            SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

def create_rag_chain(llm: ChatGroq, vectorstore: VectorStore):
    """Create a RAG (Retrieval-Augmented Generation) chain."""
    qa_prompt = create_qa_prompt()
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

def get_answer(question: str, vectorstore: VectorStore) -> str:
    """
    Generate an answer to a question using RAG with the provided vector store.

    Args:
        question: The user's question
        vectorstore: Vector store containing document embeddings

    Returns:
        str: Generated answer based on the retrieved context
    """
    llm = ChatGroq(model=DEFAULT_MODEL)
    rag_chain = create_rag_chain(llm, vectorstore)
    response = rag_chain.invoke({"input": question})
    return response['answer']
