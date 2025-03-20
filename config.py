#!/usr/bin/env python3
"""
config.py

This module defines configuration settings for the Research Assistant.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
ARCHIVE_DIR = os.path.join(BASE_DIR, "data_archive")

# Create directories if they don't exist
for directory in [DATA_DIR, CHUNKS_DIR, FAISS_INDEX_DIR, ARCHIVE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Document processing settings
DOC_PROCESSING = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "all-MiniLM-L6-v2",
}

# Retrieval settings
RETRIEVAL = {
    "search_k": 5,
    "reranking_enabled": True,
    "mmr_lambda": 0.5,  # Diversity vs. relevance tradeoff (0.0-1.0)
}

# LLM settings
LLM = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# UI settings
UI = {
    "page_title": "Research Assistant",
    "page_icon": "📚",
    "theme": "light",
}

# System prompt for the AI assistant
SYSTEM_PROMPT = """You are a research assistant helping users find information in their documents.
Base your answers solely on the context provided. If you don't know the answer based on the
retrieved context, say "I couldn't find information about that in your documents."
For each response, cite the document title and page number in [Document: Title, Page: X] format.
Keep your answers concise and focused on the question asked."""

# Function to get API key from environment or config
def get_api_key(key_name):
    """Get API key from environment variables"""
    return os.environ.get(key_name, "")
