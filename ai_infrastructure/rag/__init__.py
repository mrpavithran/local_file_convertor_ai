
"""
RAG (Retrieval-Augmented Generation) Package

A comprehensive toolkit for building RAG applications with document ingestion,
vector storage, and context retrieval capabilities.

Key Components:
- DocumentIngestor: Process PDF, DOCX, TXT, CSV files
- ChromaDBManager: Vector database operations and embeddings
- ContextRetriever: Smart context retrieval for LLM prompting
- IndexBuilder: Flexible index creation and management

Example usage:
    >>> from rag_package import DocumentIngestor, ChromaDBManager, ContextRetriever
    >>> 
    >>> # Ingest documents
    >>> ingestor = DocumentIngestor()
    >>> documents = ingestor.ingest_directory("./documents")
    >>> 
    >>> # Build index
    >>> from rag_package import IndexBuilder
    >>> builder = IndexBuilder()
    >>> result = builder.create_index(documents, "my_collection")
    >>> 
    >>> # Query with context retrieval
    >>> retriever = ContextRetriever("my_collection")
    >>> prompt = retriever.retrieve_and_format_prompt("What is machine learning?")
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core components
from .chromadb_manager import ChromaDBManager
from .document_ingestor import DocumentIngestor, ingest_documents_from_paths
from .context_retriever import ContextRetriever
from .index_builder import IndexBuilder, create_index_from_docs, rebuild_index

# Optional: Import additional utilities if available
try:
    from .advanced_retrieval import HybridRetriever, Reranker
except ImportError:
    pass

# Legacy imports for backward compatibility
__all__ = [
    # Core classes
    "ChromaDBManager",
    "DocumentIngestor", 
    "ContextRetriever",
    "IndexBuilder",
    
    # Legacy functions (maintained for backward compatibility)
    "create_index_from_docs",
    "rebuild_index", 
    "ingest_documents_from_paths",
    
    # Package metadata
    "__version__",
    "__author__", 
    "__email__",
]

# Package-level configuration
class Config:
    """Package-level configuration defaults."""
    DEFAULT_CHUNK_SIZE = 800
    DEFAULT_OVERLAP = 100
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_PERSIST_DIR = "./chroma_db"

# Convenience function for quick setup
def quick_setup(document_paths, collection_name="default", **kwargs):
    """
    Quick setup for RAG system in one function call.
    
    Args:
        document_paths: List of paths to documents
        collection_name: Name for the collection
        **kwargs: Additional arguments for indexing
    
    Returns:
        ContextRetriever instance ready for queries
    """
    from .index_builder import IndexBuilder
    
    builder = IndexBuilder(**kwargs)
    result = builder.rebuild_index(
        paths=document_paths,
        collection_name=collection_name,
        **kwargs
    )
    
    if not result["success"]:
        raise Exception(f"Failed to setup RAG system: {result.get('error', 'Unknown error')}")
    
    return ContextRetriever(collection_name=collection_name, **kwargs)

# Export config class
__all__.append("Config")
__all__.append("quick_setup")