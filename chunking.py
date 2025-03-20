#!/usr/bin/env python3
"""
chunking.py

This module provides advanced text chunking utilities for better document segmentation.
It splits text into semantically meaningful chunks for more effective retrieval.
"""

import re
import logging
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTextSplitter:
    """Enhanced text splitter with additional preprocessing and chunk metadata."""
    
    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    ):
        """
        Initialize the EnhancedTextSplitter.
        
        Args:
            chunk_size (int): Target size of each text chunk
            chunk_overlap (int): Overlap between consecutive chunks
            separators (List[str]): Separators to use for splitting, in order of preference
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before chunking for better splitting results.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Ensure proper spacing after periods for better sentence splitting
        text = re.sub(r'\.([A-Z])', '. \1', text)
        # Remove excessive newlines but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Ensure space after bullet points for better list handling
        text = re.sub(r'•(?=\S)', '• ', text)
        return text
    
    def _enhance_chunks(self, chunks: List[Document], metadata: Dict) -> List[Document]:
        """
        Enhance chunks with additional metadata and processing.
        
        Args:
            chunks (List[Document]): Original document chunks
            metadata (Dict): Original document metadata
            
        Returns:
            List[Document]: Enhanced document chunks
        """
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Create a copy of original metadata
            new_metadata = metadata.copy() if metadata else {}
            
            # Add chunk-specific metadata
            new_metadata["chunk_index"] = i
            new_metadata["chunk_count"] = len(chunks)
            
            # Add section detection heuristics
            content = chunk.page_content
            if re.search(r'^#+\s+', content):  # Heading detection
                heading_match = re.search(r'^#+\s+(.*?)$', content, re.MULTILINE)
                if heading_match:
                    new_metadata["section_heading"] = heading_match.group(1)
            
            # Create enhanced document
            enhanced_chunks.append(
                Document(page_content=content, metadata=new_metadata)
            )
        
        return enhanced_chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into semantically meaningful chunks.
        
        Args:
            documents (List[Document]): Input documents
            
        Returns:
            List[Document]: Chunked documents
        """
        all_chunks = []
        
        for doc in documents:
            # Preprocess the text
            preprocessed_text = self._preprocess_text(doc.page_content)
            
            # Create a document with preprocessed text
            preprocessed_doc = Document(
                page_content=preprocessed_text,
                metadata=doc.metadata
            )
            
            # Split the document
            chunks = self.text_splitter.split_documents([preprocessed_doc])
            
            # Enhance the chunks with additional metadata
            enhanced_chunks = self._enhance_chunks(chunks, doc.metadata)
            
            all_chunks.extend(enhanced_chunks)
        
        logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks

def get_document_splitter(
    chunk_type="default",
    chunk_size=1000,
    chunk_overlap=200
) -> EnhancedTextSplitter:
    """
    Factory function to create document splitters with different configurations.
    
    Args:
        chunk_type (str): Type of chunking strategy ("default", "small", "large")
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters
        
    Returns:
        EnhancedTextSplitter: Configured text splitter
    """
    if chunk_type == "small":
        return EnhancedTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    elif chunk_type == "large":
        return EnhancedTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )
    else:  # default
        return EnhancedTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
