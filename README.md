# Chatbot Alpha
101 chatbot project using RAG technology.

# Research Assistant Setup

This document provides step-by-step instructions for setting up the Research Assistant application.

## Prerequisites

- Python 3.9+ installed
- Pip package manager
- OpenAI API key (for chat functionality)

## Installation

1. **Clone the repository or create a new project directory**

2. **Setup a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the environment**
   
   Create a `.env` file in the project root directory with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   
   Alternatively, you can set the environment variable directly:
   ```bash
   # On Linux/Mac
   export OPENAI_API_KEY=your_openai_api_key_here
   
   # On Windows
   set OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Create required directories**
   ```bash
   mkdir -p data chunks faiss_index data_archive
   ```

## Using the Application

### Option 1: Run the Streamlit web interface

```bash
streamlit run app.py
```

This will launch the web interface where you can:
- Upload PDF documents
- Search document content
- Chat with the AI about your documents

### Option 2: Use individual components

1. **Document Ingestion**
   
   Place your PDF documents in the `data` folder, then run:
   ```bash
   python ingestion.py
   ```

2. **Document Search**
   
   To search your indexed documents:
   ```bash
   python search.py
   ```

3. **Document Chat**
   
   To chat with your documents using the Gradio interface:
   ```bash
   python chat.py
   ```

## Recommended Workflow

1. Upload your documents through the web interface or place them in the `data` folder
2. Process documents using the ingestion pipeline
3. Use the search functionality to find specific information
4. Use the chat interface for more complex research questions

## Troubleshooting

- If you encounter CUDA/GPU-related errors, edit `requirements.txt` to use `faiss-cpu` instead of `faiss-gpu`
- If document extraction seems incomplete, try different chunking settings in `config.py`
- For OpenAI API errors, verify your API key and ensure you have sufficient credits

## Advanced Usage

### Custom Embedding Models

You can modify `config.py` to use different embedding models:

```python
DOC_PROCESSING = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "your_preferred_model_name",
}
```

### Using Local LLMs

To use local language models instead of OpenAI:

1. Modify `chat.py` to use a different LLM provider
2. Update the requirements.txt with appropriate dependencies