# PDF Chat Assistant

A sophisticated conversational AI assistant that allows users to chat with their PDF documents using RAG (Retrieval-Augmented Generation) technology. Built with Streamlit, LangChain, and Groq's LLM capabilities.

## Features

- 📁 Multiple PDF document upload support
- 💬 Conversational memory with chat history
- 🔍 Advanced RAG implementation for accurate responses
- 🧠 Context-aware question reformulation
- 📊 Vector storage with ChromaDB
- 🔐 Secure API key handling
- 💻 User-friendly Streamlit interface
- 🔄 Session management for multiple conversations

## Prerequisites

- Python 3.8+
- Groq API key
- Hugging Face API token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/laxmi444/PDF-Chat-Assistant
cd PDF-Chat-Assistant
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your API keys:
```bash
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Enter your Groq API key in the interface
3. Input a session ID (or use default)
4. Upload PDF document(s)
5. Start asking questions about your documents

## How It Works

### RAG Implementation
1. **Document Processing**
   - PDFs are loaded and split into manageable chunks
   - Text is embedded using HuggingFace's all-MiniLM-L6-v2 model

2. **Vector Storage**
   - Uses ChromaDB for efficient vector storage
   - Persistent storage with customizable settings

3. **Question-Answering Pipeline**
   - Question reformulation for context awareness
   - History-aware retrieval for better context understanding
   - Concise answer generation using Groq's Gemma 2 9B model

### Components

- `PyPDFLoader`: Handles PDF document loading
- `RecursiveCharacterTextSplitter`: Splits documents into processable chunks
- `HuggingFaceEmbeddings`: Generates document embeddings
- `ChromaDB`: Vector store for efficient retrieval
- `ChatGroq`: Manages LLM interactions
- `RunnableWithMessageHistory`: Handles conversation state

## Configuration

- Model: `gemma2-9b-it`
- Chunk size: 5000 characters
- Chunk overlap: 500 characters
- Vector store: ChromaDB with persistent storage
- Embeddings: all-MiniLM-L6-v2

## Project Structure

```
.
├── app.py              # Main application file
├── .env                # Environment variables
├── chroma_db/          # Persistent vector storage
└── README.md          # Documentation
