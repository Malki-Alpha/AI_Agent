#!/usr/bin/env python3
"""
ingestion.py

This module handles the ingestion of digital PDFs. It:
- Loads PDFs from the "data" directory using PyPDFLoader.
- Treats each page as an individual Document.
- Cleans the extracted text (fixing encoding issues, normalizing Unicode,
  removing hyphenation artifacts, and filtering unwanted symbols).
- Computes embeddings using the all-MiniLM-L6-v2 model.
- Saves each processed page (with metadata) as a JSON file in the "chunks" folder.
- Creates a FAISS vector store from the precomputed embeddings and saves it locally.
"""

import os
import ftfy
import json
import re
import unicodedata
import uuid
from collections import defaultdict
import numpy as np
import faiss

from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader

# Set up logging for debugging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def advanced_clean_text(text):
    """
    Cleans text by:
    - Fixing encoding issues using ftfy.
    - Normalizing Unicode (NFKC) to combine similar characters.
    - Removing hyphenation artifacts at line breaks.
    - Replacing common unwanted symbols (e.g., bullet '•') with a neutral marker.
    - Removing extraneous symbols in the Private Use Area (e.g., '').
    - Collapsing multiple spaces and replacing newlines with spaces.
    
    Returns:
        A cleaned version of the input text.
    """
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'•', ' - ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove characters in the Private Use Area (U+E000 to U+F8FF)
    text = ''.join(ch for ch in text if not (0xE000 <= ord(ch) <= 0xF8FF))
    return text

def load_documents_from_folder(folder_path):
    """
    Loads PDF documents from a folder using PyPDFLoader.
    Each page of the PDF is treated as a separate Document with metadata.
    
    Args:
        folder_path (str): Path to the folder containing PDFs.
    
    Returns:
        List[Document]: A list of Document objects.
    """
    documents = []

    # Check if the folder exists. If not, log an error and return an empty list.
    if not os.path.exists(folder_path):
        logger.error(f"Data folder not found: {folder_path}")
        return documents
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            try:
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    # Ensure that the source is set (for metadata)
                    if "source" not in doc.metadata or not doc.metadata["source"]:
                        doc.metadata["source"] = filename
                documents.extend(loaded_docs)
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
    logger.info(f"Loaded {len(documents)} document(s).")
    return documents

def process_documents(documents, embedding_model_name="all-MiniLM-L6-v2"):
    """
    Processes each Document (each representing a PDF page) by cleaning the text,
    computing embeddings, and adding metadata.
    
    The page number is adjusted (by adding 1) if available, so that the first page is numbered 1.
    
    Args:
        documents (List[Document]): List of Documents loaded from PDFs.
        embedding_model_name (str): Name of the Hugging Face model to compute embeddings.
    
    Returns:
        List[Document]: A list of processed Document objects.
    """
    embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    processed_docs = []
    
    # Group documents by source for sequential chunk numbering per PDF
    documents_by_source = defaultdict(list)
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        documents_by_source[source].append(doc)
    
    # Process each PDF separately
    for source, docs in documents_by_source.items():
        # Sort by page number if available (assumes metadata "page" holds the page number, 0-indexed)
        docs.sort(key=lambda x: x.metadata.get("page", 0))
        
        # Extract and sanitize the base filename to use as document title
        base_filename = os.path.basename(source)
        base_name = os.path.splitext(base_filename)[0]
        sanitized_document_title = re.sub(r'[\\/*?:"<>|\']', '_', base_name)
        
        chunk_number = 1
        for doc in docs:
            # Clean the extracted text for the page
            cleaned_text = advanced_clean_text(doc.page_content)
            # Adjust page number: if present, add 1 to convert from 0-indexing
            page_number = doc.metadata.get("page", None)
            if page_number is not None:
                page_number += 1
            else:
                page_number = chunk_number  # Fallback if no page number is provided
            
            # Build metadata with document title and adjusted page number (chunk number)
            metadata = {
                "id": str(uuid.uuid4()),
                "document_title": sanitized_document_title,
                "chunk_number": page_number
            }
            # Compute embedding for the cleaned text
            embedding = embeddings_model.embed_documents([cleaned_text])[0]
            metadata["vector"] = embedding
            
            # Create a new Document with cleaned text and updated metadata
            new_doc = Document(page_content=cleaned_text, metadata=metadata)
            processed_docs.append(new_doc)
            
            # Optionally, save each processed page as a JSON file for monitoring
            save_chunk_to_json(new_doc, folder="chunks")
            chunk_number += 1
    logger.info(f"Processed {len(processed_docs)} pages (chunks).")
    return processed_docs

def save_chunk_to_json(chunk, folder="chunks"):
    """
    Saves a Document chunk to a JSON file.
    
    The filename is composed of the document title and the chunk (page) number.
    
    Args:
        chunk (Document): Document object with page_content and metadata.
        folder (str): Directory to save JSON files.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_name = f"{chunk.metadata['document_title']}_{chunk.metadata['chunk_number']}.json"
    file_path = os.path.join(folder, file_name)
    
    data = {
        "id": chunk.metadata["id"],
        "document_title": chunk.metadata["document_title"],
        "chunk_number": chunk.metadata["chunk_number"],
        "content": chunk.page_content,
        "vector": chunk.metadata.get("vector", None)
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def create_vector_store(chunks, save_path="faiss_index"):
    """
    Creates a FAISS vector store from precomputed embeddings and saves it locally.
    
    Args:
        chunks (List[Document]): List of processed Document objects.
        save_path (str): Directory where the FAISS index will be saved.
    
    Returns:
        FAISS: A FAISS vector store object.
    """
    embeddings = [chunk.metadata["vector"] for chunk in chunks]
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings))
    
    docstore = InMemoryDocstore({i: chunk for i, chunk in enumerate(chunks)})
    index_to_docstore_id = {i: i for i in range(len(chunks))}
    
    vectorstore = FAISS(
        embedding_function=None,  # Precomputed embeddings are used.
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    # Check if the save folder exists; if not, create it.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vectorstore.save_local(save_path)
    logger.info(f"FAISS vector store saved to {save_path}")
    return vectorstore

if __name__ == "__main__":
    folder_path = "data"  # Folder containing your digital PDFs
    logger.info("Loading documents from PDFs...")
    docs = load_documents_from_folder(folder_path)
    
    logger.info("Processing documents (cleaning, computing embeddings, and adding metadata)...")
    processed_docs = process_documents(docs)
    
    logger.info("Creating FAISS vector store...")
    vector_store = create_vector_store(processed_docs)
    
    logger.info("Ingestion process completed.")