#!/usr/bin/env python3
"""
search.py

This module loads the FAISS vector store created by ingestion.py and allows
the user to search for relevant content. For each search result, it prints the
document title, page (chunk) number, a snippet of the content, and the corresponding
score (distance) next to the metadata.
"""

import os
import json
import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_vector_store(load_path="faiss_index", embedding_model_name="all-MiniLM-L6-v2"):
    """
    Loads the FAISS vector store from a local directory.
    
    Args:
        load_path (str): Path to the saved FAISS index.
        embedding_model_name (str): Name of the Hugging Face model for embeddings.
    
    Returns:
        FAISS: The loaded FAISS vector store.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.load_local(
        load_path, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    logger.info(f"FAISS vector store loaded from {load_path}")
    return vectorstore

def search_vector_store(vector_store, query, top_k=5, embedding_model_name="all-MiniLM-L6-v2"):
    """
    Searches the FAISS vector store for the given query and prints the results.
    
    Each result shows:
        - Document title
        - Chunk (Page) number
        - A snippet of the content (first 200 characters)
        - The score (distance) for the result
    
    Args:
        vector_store (FAISS): The FAISS vector store.
        query (str): The search query.
        top_k (int): Number of top results to return.
        embedding_model_name (str): Name of the model for computing query embedding.
    """
    embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    query_embedding = embeddings_model.embed_documents([query])[0]
    
    # Perform the search on the FAISS index
    D, I = vector_store.index.search(np.array([query_embedding]), top_k)
    
    print("Search Results:")
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        doc_id = vector_store.index_to_docstore_id[idx]
        chunk = vector_store.docstore._dict.get(doc_id)
        if chunk:
            # Print metadata along with the score (distance)
            print(f"Document Title: {chunk.metadata['document_title']}")
            print(f"Chunk (Page) Number: {chunk.metadata['chunk_number']}")
            print(f"Score (distance): {dist:.4f}")
            snippet = chunk.page_content[:200] + ("..." if len(chunk.page_content) > 200 else "")
            print(f"Content Snippet: {snippet}")
            print("-" * 80)
    print("\n")

if __name__ == "__main__":
    vector_store = load_vector_store()
    query = input("Enter your search query: ")
    logger.info(f"Searching for: {query}")
    search_vector_store(vector_store, query)