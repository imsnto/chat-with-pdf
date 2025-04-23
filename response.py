from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from dotenv import  load_dotenv

load_dotenv()


prompt = ChatPromptTemplate(
    input_variables=["context", "input"],
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. You will answer questions based on the following context:\n\n{context}"
        ),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

def get_answer(question: str, vectorstore)-> str:
    model_name = "llama-3.1-8b-instant"
    llm = ChatGroq(model=model_name)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    response = rag_chain.invoke({"input": question})

    return response['answer']
