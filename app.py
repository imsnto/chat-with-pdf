import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from text_splitter import split_text
from embeddings import get_embeddings
from vectordb import get_vector_store
from logger import load_logger
from response import get_answer
from langchain_core.messages import AIMessage, HumanMessage

load_logger()
import logging
logger = logging.getLogger(__name__)

if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None

if 'history' not in st.session_state:
    st.session_state['history'] = []


# sidebar contents
with st.sidebar:
    st.title('Chat with PDF')
    pdf = st.file_uploader('Upload your PDF file', type='pdf')

    if pdf:
        logger.info(f'PDF uploaded: {pdf.name}')
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        if 'vector_store' not in st.session_state:
            chunks = split_text(text)
            embeddings = get_embeddings()
            st.session_state.vector_store = get_vector_store(chunks, embeddings)


def main():
    st.title('What can I help with?')
    input_text = st.chat_input('Ask question about your PDF file.')

    if input_text:
        answer = get_answer(input_text, st.session_state.vector_store)
        st.session_state['history'].append((HumanMessage(content=input_text)))
        st.session_state['history'].append((AIMessage(content=answer)))

        for message in st.session_state['history']:
            with st.chat_message(message.type):
                st.write(message.content)
        print( st.session_state['history'])



if __name__ == '__main__':
    main()