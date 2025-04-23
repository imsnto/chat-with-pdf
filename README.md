# Chat with PDF

A powerful application that allows users to have interactive conversations with their PDF documents using LangChain and Streamlit. The application uses advanced NLP techniques to process PDF content and provide relevant responses to user queries.

## Features

- 📁 PDF Document Upload
- 💬 Interactive Chat Interface
- 🔍 Semantic Search
- 🧠 Intelligent Response Generation
- 📊 Chat History Tracking
- 🚀 Vector Store Integration
- 📝 Detailed Logging

## Technology Stack

- **Frontend**: Streamlit
- **Embeddings**: HuggingFace (sentence-transformers/all-mpnet-base-v2)
- **Vector Store**: Chroma
- **PDF Processing**: PyPDF2
- **Language Models**: LangChain with Groq (llama-3.1-8b-instant)
- **Text Processing**: Custom text splitter for optimal chunk management

## Prerequisites

- Python 3.12+
- Virtual environment (recommended) 
- Groq API key

## Installation

1. Clone the repository:

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically `http://localhost:8501`)

3. Upload a PDF file using the sidebar

4. Start asking questions about your document in the chat interface

## Project Structure
├── app.py              
├── embeddings.py      
├── logger.py          
├── response.py        
├── text_splitter.py   
├── vectordb.py        
└── requirements.txt    

## Features in Detail

- **PDF Processing**: Efficiently extracts and processes text from PDF documents
- **Text Chunking**: Intelligently splits text into manageable chunks for better processing
- **Vector Storage**: Stores and retrieves document embeddings efficiently
- **Chat Interface**: User-friendly interface for document interaction
- **History Tracking**: Maintains conversation history for context
- **Logging**: Comprehensive logging for debugging and monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
