import os
import ftfy
import json
import re
from collections import defaultdict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import uuid

# Set up logging (optional, for debugging)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming this function exists for text cleaning
def advanced_clean_text(text):
    """Clean text by fixing encoding, removing hyphens at line breaks, and normalizing spaces."""
    text = ftfy.fix_text(text)
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_and_chunk_documents(documents, chunk_size=500, chunk_overlap=50, embedding_model_name="all-MiniLM-L6-v2"):
    """
    Clean and chunk documents, adding embeddings and meta-data.
    
    Args:
        documents: List of Document objects (e.g., from PyPDFLoader).
        chunk_size: Size of each chunk in characters.
        chunk_overlap: Overlap between chunks in characters.
        embedding_model_name: Name of the Hugging Face model for embeddings.
    
    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    chunked_docs = []

    # Group documents by source (e.g., cours.pdf)
    documents_by_source = defaultdict(list)
    for doc in documents:
        source = doc.metadata.get("source", "unknown")  # Use source from metadata
        documents_by_source[source].append(doc)

    # Process each source (e.g., each PDF)
    for source, docs in documents_by_source.items():
        # Sort docs by page number (if available) and concatenate text
        docs.sort(key=lambda x: x.metadata.get("page", 0))  # Ensure page order
        full_text = " ".join(doc.page_content for doc in docs)
        
        # Clean the full text
        cleaned_full_text = advanced_clean_text(full_text)
        
        # Split into chunks
        chunks = text_splitter.split_text(cleaned_full_text)
        
        # Process each chunk with sequential numbering
        chunk_number = 1
        for chunk in chunks:
            cleaned_chunk = advanced_clean_text(chunk)
            chunk_title = cleaned_chunk[:50] if len(cleaned_chunk) > 50 else cleaned_chunk
            
            # Create metadata
            metadata = {
                "id": str(uuid.uuid4()),
                "document_title": source,  # Using source as initial document title
                "chunk_title": chunk_title,
                "chunk_number": chunk_number
            }
            
            # Generate embedding
            embedding = embeddings_model.embed_documents([cleaned_chunk])[0]
            metadata["vector"] = embedding
            
            # Create Document object and save it
            chunk_doc = Document(page_content=cleaned_chunk, metadata=metadata)
            chunked_docs.append(chunk_doc)
            save_chunk_to_json(chunk_doc, folder="chunks")
            chunk_number += 1

    logger.info(f"Created {len(chunked_docs)} chunk(s). Saved as JSON files in 'chunks' folder.")
    return chunked_docs

def save_chunk_to_json(chunk, folder="chunks"):
    """
    Save a chunk to a JSON file.
    
    Args:
        chunk: Document object with page_content and metadata.
        folder: Directory to save JSON files.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Extract base filename without extension for document_title
    base_filename = os.path.basename(chunk.metadata["document_title"])  # e.g., "lhb1992026.pdf"
    base_name = os.path.splitext(base_filename)[0]  # e.g., "lhb1992026"
    sanitized_document_title = re.sub(r'[\\/*?:"<>|\']', '_', base_name)  # Sanitize (though base_name is typically clean)
    
    # Use sanitized base name and chunk number for filename
    file_name = f"{sanitized_document_title}_{chunk.metadata['chunk_number']}.json"
    file_path = os.path.join(folder, file_name)
    
    # Prepare data for JSON with updated document_title
    data = {
        "id": chunk.metadata["id"],
        "document_title": sanitized_document_title,  # Updated to base name without extension
        "chunk_title": chunk.metadata.get("chunk_title", ""),
        "chunk_number": chunk.metadata["chunk_number"],
        "content": chunk.page_content,
        "vector": chunk.metadata.get("vector", None)
    }
    
    # Write to JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_documents_from_folder(folder_path):
    """
    Load documents from a folder.
    """
    from langchain_community.document_loaders import PyPDFLoader
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            try:
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    if "source" not in doc.metadata or not doc.metadata["source"]:
                        doc.metadata["source"] = filename
                documents.extend(loaded_docs)
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
    logger.info(f"Loaded {len(documents)} document(s).")
    return documents

if __name__ == "__main__":
    folder_path = "data"  # Replace with your folder path
    logger.info("Loading documents...")
    docs = load_documents_from_folder(folder_path)
    
    logger.info("Cleaning, chunking, and computing embeddings...")
    chunked_docs = clean_and_chunk_documents(docs)
    
    logger.info("Process completed.")