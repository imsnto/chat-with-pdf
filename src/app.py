import streamlit as st
from PyPDF2 import PdfReader
from text_splitter import split_text
from embeddings import get_embeddings
from vectordb import get_vector_store
from logger import load_logger
from response import get_answer
from langchain_core.messages import AIMessage, HumanMessage
import time
import logging

class PDFChatApplication:
    """
    The PDFChatApplication class enables users to interact with PDF documents through a chat interface.
    The application parses uploaded PDF files, extracts their content, and facilitates question-answering
    about the PDFs using embeddings and vector search.

    The class provides methods to initialize session state variables, handle PDF processing, extract
    text content from PDFs, display chat history, setup a Streamlit sidebar for uploading PDFs, display
    error messages, and manage user interaction through the chat interface.
    """
    def __init__(self):
        load_logger()
        self.logger = logging.getLogger(__name__)
        self.logger.info('Application started')
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'history' not in st.session_state:
            st.session_state['history'] = []

    def process_pdf(self, pdf_file):
        text = self.extract_text_from_pdf(pdf_file)

        if 'vector_store' not in st.session_state:
            chunks = split_text(text)
            embeddings = get_embeddings()
            st.session_state['vector_store'] = get_vector_store(chunks, embeddings)

    def extract_text_from_pdf(self, pdf_file):
        """Extract text content from PDF file."""
        pdf_reader = PdfReader(pdf_file)
        return "".join(page.extract_text() for page in pdf_reader.pages)

    def setup_sidebar(self):
        """Setup the sidebar with PDF upload functionality."""
        with st.sidebar:
            st.title('Chat with PDF')
            pdf = st.file_uploader('Upload your PDF file', type='pdf')

            if pdf:
                self.process_pdf(pdf)

    def display_chat_history(self, history):
        """Display all messages from chat history."""
        for message in history:
            with st.chat_message(message.type):
                st.write(message.content)

    def show_error_message(self, error_message):
        """Display error message when pdf is not uploaded."""
        with st.spinner():
            time.sleep(1)
            st.error(error_message)

    def handle_user_input(self, user_input):
        try:
            if 'vector_store' in st.session_state:
                answer = get_answer(user_input, st.session_state['vector_store'])
                st.session_state['history'].append((HumanMessage(content=user_input)))
                st.session_state['history'].append((AIMessage(content=answer)))

                self.display_chat_history(st.session_state['history'][-2:])
            else:
                self.show_error_message('There is no data.')
        except AttributeError as e:
            self.show_error_message('Please upload a PDF file.')

    def run(self):
        st.title('What can I help with?')
        user_input = st.chat_input('Ask question about your PDF file.')
        self.display_chat_history(st.session_state['history'])

        if user_input:
            self.handle_user_input(user_input)

def main():
    app = PDFChatApplication()
    app.setup_sidebar()
    app.run()

if __name__ == '__main__':
    main()