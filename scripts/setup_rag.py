#!/usr/bin/env python3
"""
RAG (Retrieval Augmented Generation) system setup script.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table


def setup_rag_system(embedding_model: str = "all-MiniLM-L6-v2", 
                    vector_store: str = "chroma",
                    reset: bool = False) -> bool:
    """
    Setup RAG system for AI File System.
    
    Args:
        embedding_model: Embedding model to use
        vector_store: Vector store type (chroma, faiss, etc.)
        reset: Whether to reset existing RAG system
        
    Returns:
        True if setup successful
    """
    console = Console()
    
    console.print(Panel.fit(
        "🧠 RAG System Setup for AI File System",
        style="bold blue"
    ))
    
    try:
        # Create necessary directories
        base_dir = Path(__file__).parent.parent
        rag_dirs = [
            base_dir / "data" / "chroma_db",
            base_dir / "data" / "chroma_db" / "embeddings",
            base_dir / "data" / "chroma_db" / "metadata",
            base_dir / "ai_infrastructure" / "rag" / "vector_stores",
            base_dir / "ai_infrastructure" / "rag" / "documents"
        ]
        
        console.print("📁 Creating directory structure...")
        for directory in rag_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            console.print(f"    Created: {directory}")
        
        # Reset if requested
        if reset:
            console.print("🔄 Resetting existing RAG system...")
            reset_rag_system()
        
        # Install required packages
        console.print("📦 Installing RAG dependencies...")
        if not install_rag_dependencies():
            console.print("❌ Failed to install RAG dependencies")
            return False
        
        # Initialize vector store
        console.print("🗄️ Initializing vector store...")
        if not initialize_vector_store(vector_store, embedding_model):
            console.print("❌ Failed to initialize vector store")
            return False
        
        # Create configuration
        console.print("⚙️ Creating RAG configuration...")
        if not create_rag_config(embedding_model, vector_store):
            console.print("❌ Failed to create RAG configuration")
            return False
        
        # Test RAG system
        console.print("🧪 Testing RAG system...")
        if not test_rag_system():
            console.print("❌ RAG system test failed")
            return False
        
        console.print("✅ RAG system setup completed successfully!")
        return True
        
    except Exception as e:
        console.print(f"❌ RAG setup failed: {e}")
        return False


def install_rag_dependencies() -> bool:
    """Install RAG system dependencies."""
    console = Console()
    
    rag_packages = [
        "chromadb",
        "sentence-transformers",
        "llama-index",
        "pypdf2",
        "python-docx",
        "beautifulsoup4"
    ]
    
    try:
        import importlib
        import subprocess
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Installing packages...", total=len(rag_packages))
            
            for package in rag_packages:
                progress.update(task, description=f"Installing {package}...")
                
                try:
                    # Try to import first to check if already installed
                    importlib.import_module(package.replace("-", "_"))
                    progress.update(task, advance=1)
                    continue
                except ImportError:
                    # Install using pip
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", package
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        progress.update(task, advance=1)
                    else:
                        console.print(f"    ❌ Failed to install {package}: {result.stderr}")
                        return False
        
        return True
        
    except Exception as e:
        console.print(f"❌ Dependency installation failed: {e}")
        return False


def initialize_vector_store(vector_store: str, embedding_model: str) -> bool:
    """Initialize the vector store."""
    console = Console()
    
    try:
        if vector_store == "chroma":
            return initialize_chroma_db(embedding_model)
        else:
            console.print(f"❌ Unsupported vector store: {vector_store}")
            return False
            
    except Exception as e:
        console.print(f"❌ Vector store initialization failed: {e}")
        return False


def initialize_chroma_db(embedding_model: str) -> bool:
    """Initialize ChromaDB vector store."""
    console = Console()
    
    try:
        # Import ChromaDB
        import chromadb
        
        # Initialize persistent client
        db_path = Path(__file__).parent.parent / "data" / "chroma_db"
        client = chromadb.PersistentClient(path=str(db_path))
        
        # Create collections for different document types
        collections = ["documents", "code", "images_metadata", "general"]
        
        for collection_name in collections:
            try:
                collection = client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                console.print(f"    ✅ Created collection: {collection_name}")
            except Exception as e:
                console.print(f"    ⚠️ Collection {collection_name} may already exist: {e}")
        
        # Test the connection
        test_collection = client.get_collection("documents")
        test_collection.add(
            documents=["This is a test document for RAG system initialization."],
            metadatas=[{"type": "test", "source": "setup"}],
            ids=["test_id_001"]
        )
        
        # Verify we can query
        results = test_collection.query(
            query_texts=["test document"],
            n_results=1
        )
        
        if results['documents']:
            console.print("    ✅ ChromaDB initialized and tested successfully")
            return True
        else:
            console.print("    ❌ ChromaDB test query failed")
            return False
            
    except Exception as e:
        console.print(f"❌ ChromaDB initialization failed: {e}")
        return False


def create_rag_config(embedding_model: str, vector_store: str) -> bool:
    """Create RAG configuration file."""
    console = Console()
    
    try:
        config = {
            'rag_system': {
                'embedding_model': embedding_model,
                'vector_store': vector_store,
                'collections': {
                    'documents': 'General document storage',
                    'code': 'Source code and programming content',
                    'images_metadata': 'Image descriptions and metadata',
                    'general': 'Miscellaneous content'
                },
                'settings': {
                    'similarity_threshold': 0.7,
                    'max_results': 10,
                    'chunk_size': 1000,
                    'chunk_overlap': 200
                }
            }
        }
        
        config_path = Path(__file__).parent.parent / "ai_infrastructure" / "rag" / "rag_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        console.print(f"✅ RAG configuration saved: {config_path}")
        return True
        
    except Exception as e:
        console.print(f"❌ Failed to create RAG configuration: {e}")
        return False


def test_rag_system() -> bool:
    """Test the RAG system functionality."""
    console = Console()
    
    try:
        # Test basic imports
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        # Test embedding model
        console.print("    Testing embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_embedding = model.encode(["test document"])
        
        if len(test_embedding) > 0:
            console.print("    ✅ Embedding model working")
        else:
            console.print("    ❌ Embedding model failed")
            return False
        
        # Test ChromaDB connection
        console.print("    Testing vector store...")
        db_path = Path(__file__).parent.parent / "data" / "chroma_db"
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection("documents")
        
        # Test query
        results = collection.query(
            query_texts=["test"],
            n_results=1
        )
        
        console.print("    ✅ Vector store working")
        return True
        
    except Exception as e:
        console.print(f"❌ RAG system test failed: {e}")
        return False


def reset_rag_system():
    """Reset the RAG system by clearing all data."""
    console = Console()
    
    try:
        rag_dirs = [
            Path(__file__).parent.parent / "data" / "chroma_db",
            Path(__file__).parent.parent / "ai_infrastructure" / "rag" / "vector_stores"
        ]
        
        for directory in rag_dirs:
            if directory.exists():
                shutil.rmtree(directory)
                console.print(f"    Cleared: {directory}")
        
        console.print("✅ RAG system reset complete")
        
    except Exception as e:
        console.print(f"⚠️  Partial reset: {e}")


def get_rag_status() -> Dict[str, any]:
    """Get current RAG system status."""
    status = {
        'vector_store': 'unknown',
        'embedding_model': 'unknown',
        'collections': [],
        'document_count': 0
    }
    
    try:
        import chromadb
        
        db_path = Path(__file__).parent.parent / "data" / "chroma_db"
        if db_path.exists():
            client = chromadb.PersistentClient(path=str(db_path))
            collections = client.list_collections()
            
            status['vector_store'] = 'chroma'
            status['collections'] = [col.name for col in collections]
            
            # Count documents in main collection
            if collections:
                main_collection = client.get_collection("documents")
                status['document_count'] = main_collection.count()
        
        # Check embedding model
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            status['embedding_model'] = 'all-MiniLM-L6-v2'
        except:
            status['embedding_model'] = 'not_loaded'
    
    except Exception as e:
        status['error'] = str(e)
    
    return status


def main():
    """Main function for standalone execution."""
    console = Console()
    
    parser = argparse.ArgumentParser(description="Setup RAG system for AI File System")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", 
                       help="Embedding model to use")
    parser.add_argument("--vector-store", default="chroma", 
                       choices=["chroma", "faiss"], help="Vector store type")
    parser.add_argument("--reset", action="store_true", 
                       help="Reset existing RAG system")
    parser.add_argument("--status", action="store_true", 
                       help="Show current RAG system status")
    
    args = parser.parse_args()
    
    if args.status:
        status = get_rag_status()
        
        table = Table(title="RAG System Status")
        table.add_column("Component", style="bold")
        table.add_column("Status", style="green")
        
        table.add_row("Vector Store", status['vector_store'])
        table.add_row("Embedding Model", status['embedding_model'])
        table.add_row("Collections", ", ".join(status['collections']))
        table.add_row("Document Count", str(status['document_count']))
        
        if 'error' in status:
            table.add_row("Error", status['error'], style="red")
        
        console.print(table)
        return
    
    success = setup_rag_system(
        embedding_model=args.embedding_model,
        vector_store=args.vector_store,
        reset=args.reset
    )
    
    if success:
        console.print("🎉 RAG system setup completed successfully!")
        sys.exit(0)
    else:
        console.print("💥 RAG system setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    main()