from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Constants
DEFAULT_MODEL = "llama-3.1-8b-instant"
SYSTEM_TEMPLATE = ("You are a helpful assistant. You will answer questions based on "
                   "the following context:\n\n{context}")
rag_qa_template = """
            User Query: "{input}"

            Based on the conversation history and retrieved context below, provide a natural, engaging, 
            and concise response to the user's query. Ensure the answer is relevant, incorporates 
            available context, and aligns with the instructions. If the context is insufficient, 
            politely explain and offer to clarify or search for additional details.

            Retrieved Context: {context}
            Conversation History: {conversation_history}
            Response:
"""

qa_template = """
            User Query: "{input}"

            Respond to the user's query in a friendly, concise, and natural way, as if you're having a casual 
            conversation. Use the conversation history to keep the chat consistent and relevant. If no history is
            available, answer based on your knowledge. If the query is unclear, ask a clarifying question politely.

            Conversation History: {conversation_history}

            Response:
"""

def create_qa_prompt() -> ChatPromptTemplate:
    """Create the question-answering prompt template."""
    return PromptTemplate.from_template(
        template=rag_qa_template
    )

def create_rag_chain(llm: ChatGroq, vectorstore: VectorStore):
    """Create a RAG (Retrieval-Augmented Generation) chain."""
    qa_prompt = create_qa_prompt()
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

def get_answer(question: str, vectorstore: VectorStore, conversation_history) -> str:
    """
    Generate an answer to a question using RAG with the provided vector store.

    Args:
        question: The user's question
        vectorstore: Vector store containing document embeddings

    Returns:
        str: Generated answer based on the retrieved context
    """
    print('get answer called')

    llm = ChatGroq(model=DEFAULT_MODEL)
    rag_chain = create_rag_chain(llm, vectorstore)
    response = rag_chain.invoke({"input": question, "conversation_history":conversation_history})
    return response['answer']

def get_answer_without_context(question: str, conversation_history):
    prompt = PromptTemplate.from_template(
        template=qa_template
    )

    llm = ChatGroq(model=DEFAULT_MODEL)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"input": question, "conversation_history": conversation_history})
    return response