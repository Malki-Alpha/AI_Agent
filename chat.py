#!/usr/bin/env python3
"""
chat.py

This module provides a chat interface for interacting with documents loaded into the RAG system.
It combines vector search retrieval with an LLM to generate contextual responses based on
document content.
"""

import os
import logging
from typing import List, Dict, Any

import gradio as gr
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Import local modules
from search import load_vector_store

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompt for the AI assistant
SYSTEM_PROMPT = """You are a research assistant helping users find information in their documents.
Base your answers solely on the context provided. If you don't know the answer based on the
retrieved context, say "I couldn't find information about that in your documents."
For each response, cite the document title and page number in [Document: Title, Page: X] format.
Keep your answers concise and focused on the question asked."""

# Template for retrieval question prompting
QUESTION_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say "I don't have enough information to answer that question."
Don't try to make up an answer.

{context}

Question: {question}
Conversation history: {chat_history}
"""

def setup_chat_agent(vector_store_path="faiss_index"):
    """
    Sets up a conversational agent using the vector store for document retrieval.
    
    Args:
        vector_store_path (str): Path to the FAISS vector store
        
    Returns:
        ConversationalRetrievalChain: A chain that combines retrieval with conversation
    """
    # Load the vector store
    vector_store = load_vector_store(load_path=vector_store_path)
    
    # Set up the language model (requires OpenAI API key in environment)
    # You can replace this with a different model as needed
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",  # Updated for gpt-4o-mini
        temperature=0.3,
    )
    
    # Set up conversation memory with explicit output_key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",  # This is the critical fix
        return_messages=True
    )
    
    # Create the question prompt
    question_prompt = PromptTemplate(
        template=QUESTION_TEMPLATE,
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
    
    return qa_chain

def format_response_with_sources(response):
    """
    Formats the response with source citations from the retrieved documents.
    
    Args:
        response (dict): Response from the ConversationalRetrievalChain
        
    Returns:
        str: Formatted response with source citations
    """
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

def chat_interface():
    """
    Creates a Gradio interface for chatting with documents.
    """
    # Set up the chat agent
    qa_chain = setup_chat_agent()
    
    def respond(message, history):
        """Process user message and generate response"""
        try:
            response = qa_chain({"question": message})
            formatted_response = format_response_with_sources(response)
            return formatted_response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while trying to answer: {str(e)}"
    
    # Create the Gradio interface
    demo = gr.ChatInterface(
        respond,
        title="Document Research Assistant",
        description="Ask questions about your uploaded PDFs",
        theme="soft",
    )
    
    demo.launch(share=True)

if __name__ == "__main__":
    # Check if OPENAI_API_KEY is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set. Please set it before running this script.")
        print("You can set it temporarily with: export OPENAI_API_KEY=your_key_here")
        exit(1)
    
    chat_interface()