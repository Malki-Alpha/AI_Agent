#!/usr/bin/env python3
"""
app.py

This module creates a Streamlit web interface for the Research Assistant.
It provides three main functions:
1. Document upload and ingestion
2. Document search
3. AI chat based on document content
"""

import os
import sys
import tempfile
import logging
import time
from pathlib import Path

import streamlit as st
import numpy as np

# Import local modules (adjust imports if needed)
from ingestion import load_documents_from_folder, process_documents, create_vector_store
from search import load_vector_store, search_vector_store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load or create an OpenAI API key
def get_openai_api_key():
    if "OPENAI_API_KEY" in st.session_state:
        return st.session_state["OPENAI_API_KEY"]
    
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        st.session_state["OPENAI_API_KEY"] = api_key
    
    return api_key

# Initialize session state variables
def init_session_state():
    session_vars = [
        "messages", 
        "vector_store",
        "document_list",
        "processing_complete",
        "qa_chain"
    ]
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None
    
    # Initialize messages if not already done
    if st.session_state["messages"] is None:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I'm your research assistant. Upload documents, search content, or ask questions about your documents."}]

# Set up the sidebar
def setup_sidebar():
    st.sidebar.title("Research Assistant")
    st.sidebar.markdown("---")
    
    # API Key management
    with st.sidebar.expander("API Key Settings", expanded=False):
        api_key = st.text_input("OpenAI API Key", value=get_openai_api_key(), type="password")
        if api_key:
            st.session_state["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Upload Documents", "Search Documents", "Chat Assistant"])
    
    st.sidebar.markdown("---")
    
    # About section
    with st.sidebar.expander("About"):
        st.markdown("""
        # RAG Research Assistant
        
        This application helps you analyze documents through:
        - Document ingestion and preprocessing
        - Semantic search across your documents
        - AI-powered chat to answer questions about your documents
        
        Built with LangChain, FAISS, and HuggingFace.
        """)
    
    return page

# Function to handle document uploads
def upload_documents():
    st.title("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner("Processing uploaded documents..."):
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            data_dir = os.path.join(temp_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Save uploaded files to temp directory
            for uploaded_file in uploaded_files:
                file_path = os.path.join(data_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Process the documents
            docs = load_documents_from_folder(data_dir)
            if not docs:
                st.error("No valid documents were found in the upload.")
                return
            
            st.info(f"Loaded {len(docs)} pages from {len(uploaded_files)} documents.")
            
            # Process documents and create vector store
            processed_docs = process_documents(docs)
            vector_store = create_vector_store(processed_docs)
            
            # Update session state
            st.session_state["vector_store"] = vector_store
            st.session_state["document_list"] = [doc.metadata["document_title"] for doc in processed_docs]
            st.session_state["processing_complete"] = True
            
            st.success("Documents processed and indexed successfully!")
            
            # Display document information
            display_document_info(processed_docs)

# Function to display document information
def display_document_info(processed_docs):
    doc_titles = {}
    for doc in processed_docs:
        title = doc.metadata["document_title"]
        if title in doc_titles:
            doc_titles[title] += 1
        else:
            doc_titles[title] = 1
    
    st.subheader("Processed Documents")
    for title, count in doc_titles.items():
        st.markdown(f"- **{title}**: {count} pages")

# Function to handle document search
def search_documents():
    st.title("Search Documents")
    
    if not st.session_state.get("processing_complete"):
        if os.path.exists("faiss_index"):
            try:
                vector_store = load_vector_store()
                st.session_state["vector_store"] = vector_store
                st.session_state["processing_complete"] = True
            except Exception as e:
                st.warning("Please upload documents first before searching.")
                return
        else:
            st.warning("Please upload documents first before searching.")
            return
    
    query = st.text_input("Enter your search query:")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    if query and st.button("Search"):
        with st.spinner("Searching..."):
            # Get embedding model
            embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            query_embedding = embeddings_model.embed_documents([query])[0]
            
            # Perform search
            D, I = st.session_state["vector_store"].index.search(np.array([query_embedding]), top_k)
            
            st.subheader("Search Results")
            
            for dist, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                
                doc_id = st.session_state["vector_store"].index_to_docstore_id[idx]
                chunk = st.session_state["vector_store"].docstore._dict.get(doc_id)
                
                if chunk:
                    with st.expander(f"{chunk.metadata['document_title']} - Page {chunk.metadata['chunk_number']} (Score: {dist:.4f})"):
                        st.markdown(chunk.page_content)

# Function to set up QA chain if not already done
def setup_qa_chain():
    if st.session_state["qa_chain"] is None and st.session_state.get("processing_complete"):
        try:
            # Import here to avoid circular imports
            from langchain.chains import ConversationalRetrievalChain
            from langchain_openai import ChatOpenAI
            from langchain.memory import ConversationBufferMemory
            from langchain.prompts import PromptTemplate
            from search import load_vector_store
            
            # Load vector store
            vector_store = st.session_state["vector_store"] or load_vector_store()
            
            # Set up language model
            llm = ChatOpenAI(
                model_name="gpt-4o-mini",  # Updated for gpt-4o-mini
                temperature=0.3,
            )
            
            # Set up conversation memory with explicit output_key
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",  # This fixes the error
                return_messages=True
            )
            
            # Template for retrieval question prompting
            question_template = """
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say "I don't have enough information to answer that question."
            Don't try to make up an answer.

            {context}

            Question: {question}
            Conversation history: {chat_history}
            """
            
            # Create the question prompt
            question_prompt = PromptTemplate(
                template=question_template,
                input_variables=["context", "question", "chat_history"]
            )
            
            # Create the conversational chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": question_prompt},
                return_source_documents=True
            )
            
            # Define formatter function
            def format_response_with_sources(response):
                """Format response with sources"""
                answer = response["answer"]
                source_documents = response["source_documents"]
                
                # If there are no explicit citations in the answer,
                # add sources at the end of the response
                if not any(f"[Document:" in answer for doc in source_documents):
                    sources_text = "\n\nSources:\n"
                    seen_sources = set()
                    
                    for doc in source_documents:
                        doc_title = doc.metadata.get("document_title", "Unknown")
                        chunk_number = doc.metadata.get("chunk_number", "Unknown")
                        source_key = f"{doc_title}-{chunk_number}"
                        
                        if source_key not in seen_sources:
                            sources_text += f"- {doc_title}, Page {chunk_number}\n"
                            seen_sources.add(source_key)
                    
                    return answer + sources_text
                
                return answer
            
            # Store both the chain and the formatter function
            st.session_state["qa_chain"] = {
                "chain": qa_chain,
                "format": format_response_with_sources
            }
        except Exception as e:
            st.error(f"Error setting up chat agent: {e}")
            if "OPENAI_API_KEY" not in os.environ:
                st.warning("OpenAI API key not found. Please add it in the sidebar.")

# Function to handle chat interface
def chat_interface():
    st.title("Chat with Your Documents")
    
    # Check if documents are processed
    if not st.session_state.get("processing_complete"):
        if os.path.exists("faiss_index"):
            try:
                vector_store = load_vector_store()
                st.session_state["vector_store"] = vector_store
                st.session_state["processing_complete"] = True
            except Exception as e:
                st.warning("Please upload and process documents before using the chat interface.")
                return
        else:
            st.warning("Please upload and process documents before using the chat interface.")
            return
    
    # Set up QA chain if needed
    setup_qa_chain()
    
    # Check for API key
    if not get_openai_api_key():
        st.warning("Please add your OpenAI API key in the sidebar to use the chat functionality.")
        return
    
    # Display chat messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Get response from QA chain
                qa_chain = st.session_state["qa_chain"]["chain"]
                format_func = st.session_state["qa_chain"]["format"]
                
                response = qa_chain({"question": prompt})
                formatted_response = format_func(response)
                
                # Update placeholder with response
                message_placeholder.markdown(formatted_response)
                
                # Add assistant message to chat history
                st.session_state["messages"].append({"role": "assistant", "content": formatted_response})
                
            except Exception as e:
                message_placeholder.markdown(f"Error generating response: {str(e)}")
                st.session_state["messages"].append({"role": "assistant", "content": f"Error generating response: {str(e)}"})

# Main function
def main():
    # Initialize session state
    init_session_state()
    
    # Set up sidebar and get selected page
    page = setup_sidebar()
    
    # Display selected page
    if page == "Upload Documents":
        upload_documents()
    elif page == "Search Documents":
        search_documents()
    elif page == "Chat Assistant":
        chat_interface()

if __name__ == "__main__":
    main()