"""
ChromaDB manager: handles embeddings (sentence-transformers) and chroma collections.
Dependencies: chromadb, sentence-transformers
"""
from typing import List, Dict, Optional, Union, Any
import os
import uuid
import logging
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from chromadb.api.types import Where, WhereDocument
from sentence_transformers import SentenceTransformer

# Set up logging
logger = logging.getLogger(__name__)

class ChromaDBManager:
    def __init__(
        self, 
        persist_directory: str = "./chroma_db", 
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_model_kwargs: Optional[Dict[str, Any]] = None,
        chroma_settings: Optional[Dict[str, Any]] = None
    ):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with configurable settings
        default_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        )
        
        if chroma_settings:
            for key, value in chroma_settings.items():
                setattr(default_settings, key, value)
                
        self.client = chromadb.Client(default_settings)
        
        # Initialize embedding model with configurable parameters
        model_kwargs = embedding_model_kwargs or {}
        self.model = SentenceTransformer(embedding_model_name, **model_kwargs)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Initialized ChromaDBManager with model: {embedding_model_name}")

    def get_or_create_collection(self, name: str, metadata: Optional[Dict] = None) -> chromadb.Collection:
        """Get existing collection or create new one with metadata."""
        try:
            collection = self.client.get_collection(name)
            logger.debug(f"Retrieved existing collection: {name}")
            return collection
        except Exception as e:
            collection_metadata = metadata or {"hnsw:space": "cosine"}
            collection = self.client.create_collection(
                name=name,
                metadata=collection_metadata
            )
            logger.info(f"Created new collection: {name}")
            return collection

    def list_collections(self) -> List[str]:
        """List all available collections."""
        collections = self.client.list_collections()
        return [collection.name for collection in collections]

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.warning(f"Collection {name} not found or couldn't be deleted: {e}")
            return False

    def embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> List[List[float]]:
        """Generate embeddings for texts with batching and progress tracking."""
        if not texts:
            return []
        
        # Use tqdm for progress bar if requested
        iterable = texts
        if show_progress and len(texts) > batch_size:
            iterable = tqdm(texts, desc="Generating embeddings")
        
        # Encode in batches for better performance
        embeddings = self.model.encode(
            iterable,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        
        # Convert to list of lists
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        return [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]

    def upsert_documents(
        self, 
        collection_name: str, 
        docs: List[Dict],
        batch_size: int = 100,
        show_progress: bool = False
    ) -> List[str]:
        """Upsert documents with batching support."""
        if not docs:
            return []
        
        col = self.get_or_create_collection(collection_name)
        all_ids = []
        
        # Process in batches
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_ids, batch_metadatas, batch_texts = [], [], []
            
            for doc in batch_docs:
                batch_ids.append(doc.get("id", str(uuid.uuid4())))
                batch_texts.append(doc["text"])
                batch_metadatas.append(doc.get("metadata", {}))
            
            # Generate embeddings for the batch
            batch_embeddings = self.embeddings(batch_texts, show_progress=False)
            
            # Upsert to collection
            col.upsert(
                ids=batch_ids,
                metadatas=batch_metadatas,
                documents=batch_texts,
                embeddings=batch_embeddings
            )
            
            all_ids.extend(batch_ids)
            
            if show_progress:
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1}")
        
        logger.info(f"Upserted {len(all_ids)} documents to collection '{collection_name}'")
        return all_ids

    def query(
        self, 
        collection_name: str, 
        query: str, 
        top_k: int = 5,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None,
        include: List[str] = ["metadatas", "documents", "distances"]
    ) -> List[Dict]:
        """Query collection with flexible filtering options."""
        col = self.get_or_create_collection(collection_name)
        
        # Generate query embedding
        query_embedding = self.embeddings([query])[0]
        
        # Perform query
        result = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=include
        )
        
        hits = []
        if result["documents"] and result["documents"][0]:
            for idx, doc in enumerate(result["documents"][0]):
                hit = {
                    "id": result["ids"][0][idx],
                    "document": doc,
                    "metadata": result.get("metadatas", [[]])[0][idx] if "metadatas" in include else {},
                    "distance": result.get("distances", [[]])[0][idx] if "distances" in include else None,
                    "similarity": 1 - result.get("distances", [[]])[0][idx] if "distances" in include else None
                }
                hits.append(hit)
        
        logger.debug(f"Query returned {len(hits)} results from collection '{collection_name}'")
        return hits

    def search_by_embedding(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Where] = None
    ) -> List[Dict]:
        """Search using pre-computed embeddings."""
        col = self.get_or_create_collection(collection_name)
        
        result = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"]
        )
        
        hits = []
        if result["documents"] and result["documents"][0]:
            for idx, doc in enumerate(result["documents"][0]):
                hit = {
                    "id": result["ids"][0][idx],
                    "document": doc,
                    "metadata": result["metadatas"][0][idx],
                    "distance": result["distances"][0][idx],
                    "similarity": 1 - result["distances"][0][idx]
                }
                hits.append(hit)
        
        return hits

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a collection."""
        try:
            col = self.get_or_create_collection(collection_name)
            count = col.count()
            
            # Get sample of metadata keys
            sample_result = col.get(limit=1)
            metadata_keys = set()
            if sample_result["metadatas"]:
                for metadata in sample_result["metadatas"]:
                    metadata_keys.update(metadata.keys())
            
            return {
                "collection_name": collection_name,
                "document_count": count,
                "embedding_dimension": self.embedding_dimension,
                "metadata_fields": list(metadata_keys)
            }
        except Exception as e:
            logger.error(f"Error getting stats for collection {collection_name}: {e}")
            return {}

    def persist(self) -> bool:
        """Persist data to disk."""
        try:
            self.client.persist()
            logger.debug("Data persisted successfully")
            return True
        except Exception as e:
            logger.error(f"Error persisting data: {e}")
            return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.persist()

# Example usage
if __name__ == "__main__":
    # Initialize with custom settings
    manager = ChromaDBManager(
        embedding_model_name="all-MiniLM-L6-v2",
        persist_directory="./my_chroma_db"
    )
    
    # Example documents
    docs = [
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"topic": "AI", "source": "textbook"}
        },
        {
            "text": "Deep learning uses neural networks with multiple layers.",
            "metadata": {"topic": "AI", "source": "research_paper"}
        }
    ]
    
    # Upsert with progress tracking
    ids = manager.upsert_documents("ai_docs", docs, show_progress=True)
    
    # Query with metadata filtering
    results = manager.query(
        "ai_docs",
        "What is machine learning?",
        top_k=3,
        where={"topic": "AI"}
    )
    
    # Get collection statistics
    stats = manager.get_collection_stats("ai_docs")
    print(f"Collection stats: {stats}")