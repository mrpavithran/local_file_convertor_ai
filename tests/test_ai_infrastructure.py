"""
Tests for AI infrastructure components including Ollama, RAG, and MCP.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_infrastructure.ollama.model_manager import ModelManager
from ai_infrastructure.ollama.prompt_executor import PromptExecutor
from ai_infrastructure.rag.document_ingestor import DocumentIngestor
from ai_infrastructure.rag.context_retriever import ContextRetriever
from ai_infrastructure.rag.chromadb_manager import ChromaDBManager
from ai_infrastructure.mcp.mcp_server import MCPServer
from ai_infrastructure.mcp.tool_registry import ToolRegistry


class TestOllamaInfrastructure:
    """Test Ollama model management and prompt execution."""
    
    @pytest.fixture
    def model_manager(self):
        """Create a ModelManager instance for testing."""
        return ModelManager()
    
    @pytest.fixture
    def prompt_executor(self):
        """Create a PromptExecutor instance for testing."""
        return PromptExecutor()
    
    def test_model_manager_initialization(self, model_manager):
        """Test ModelManager initialization."""
        assert model_manager is not None
        assert hasattr(model_manager, 'base_url')
        assert hasattr(model_manager, 'default_model')
    
    @patch('requests.get')
    def test_check_ollama_running(self, mock_get, model_manager):
        """Test checking if Ollama is running."""
        # Mock successful response
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {'models': []}
        
        assert model_manager.check_ollama_running() is True
        
        # Mock failed response
        mock_get.return_value.status_code = 500
        assert model_manager.check_ollama_running() is False
    
    @patch('requests.post')
    def test_generate_completion(self, mock_post, prompt_executor):
        """Test generating AI completions."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'model': 'test-model',
            'response': 'Test completion response'
        }
        mock_post.return_value = mock_response
        
        result = prompt_executor.generate_completion(
            prompt="Test prompt",
            model="test-model"
        )
        
        assert result is not None
        assert 'response' in result
    
    def test_prompt_template_loading(self, prompt_executor):
        """Test loading prompt templates."""
        # Test with valid template
        template = prompt_executor.load_prompt_template('conversion', 'general_conversion')
        assert template is not None
        assert 'system' in template
        assert 'user' in template
        
        # Test with invalid template
        with pytest.raises(ValueError):
            prompt_executor.load_prompt_template('invalid', 'invalid')


class TestRAGInfrastructure:
    """Test RAG system components."""
    
    @pytest.fixture
    def chroma_manager(self):
        """Create a ChromaDBManager instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_chroma"
            manager = ChromaDBManager(str(db_path))
            yield manager
    
    @pytest.fixture
    def document_ingestor(self):
        """Create a DocumentIngestor instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_chroma"
            ingestor = DocumentIngestor(str(db_path))
            yield ingestor
    
    @pytest.fixture
    def context_retriever(self):
        """Create a ContextRetriever instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_chroma"
            retriever = ContextRetriever(str(db_path))
            yield retriever
    
    def test_chromadb_initialization(self, chroma_manager):
        """Test ChromaDB initialization."""
        assert chroma_manager.client is not None
        assert chroma_manager.db_path is not None
        
        # Test collection creation
        collection = chroma_manager.get_collection("test_documents")
        assert collection is not None
    
    def test_document_ingestion(self, document_ingestor):
        """Test document ingestion functionality."""
        # Create a test document
        test_doc_path = Path(__file__).parent / "test_data" / "sample_documents" / "test.txt"
        test_doc_path.parent.mkdir(parents=True, exist_ok=True)
        test_doc_path.write_text("This is a test document for RAG ingestion.")
        
        # Ingest document
        result = document_ingestor.ingest_document(str(test_doc_path))
        assert result['success'] is True
        assert 'document_id' in result
        
        # Cleanup
        test_doc_path.unlink()
    
    def test_context_retrieval(self, context_retriever, document_ingestor):
        """Test context retrieval functionality."""
        # First ingest a test document
        test_doc_path = Path(__file__).parent / "test_data" / "sample_documents" / "test_retrieval.txt"
        test_doc_path.parent.mkdir(parents=True, exist_ok=True)
        test_doc_path.write_text("""
        Artificial intelligence is transforming how we work with files.
        AI-powered file systems can automatically categorize, convert, and enhance documents.
        Machine learning models understand content and context to provide intelligent file operations.
        """)
        
        document_ingestor.ingest_document(str(test_doc_path))
        
        # Retrieve context
        results = context_retriever.retrieve(
            query="artificial intelligence file systems",
            n_results=2
        )
        
        assert results is not None
        assert len(results['documents']) > 0
        assert 'artificial intelligence' in results['documents'][0][0].lower()
        
        # Cleanup
        test_doc_path.unlink()
    
    def test_multiple_collections(self, chroma_manager):
        """Test working with multiple collections."""
        collections = ["documents", "code", "images_metadata"]
        
        for collection_name in collections:
            collection = chroma_manager.get_collection(collection_name)
            assert collection is not None
            
            # Test adding and querying
            collection.add(
                documents=[f"Test document for {collection_name}"],
                metadatas=[{"type": "test", "source": "pytest"}],
                ids=[f"test_{collection_name}_001"]
            )
            
            results = collection.query(
                query_texts=["test document"],
                n_results=1
            )
            
            assert len(results['documents']) > 0


class TestMCPInfrastructure:
    """Test MCP server and tool registry."""
    
    @pytest.fixture
    def tool_registry(self):
        """Create a ToolRegistry instance for testing."""
        registry = ToolRegistry()
        return registry
    
    @pytest.fixture
    def mcp_server(self):
        """Create an MCPServer instance for testing."""
        server = MCPServer()
        return server
    
    def test_tool_registry_initialization(self, tool_registry):
        """Test ToolRegistry initialization."""
        assert tool_registry is not None
        assert hasattr(tool_registry, 'tools')
        assert isinstance(tool_registry.tools, dict)
    
    def test_tool_discovery(self, tool_registry):
        """Test tool discovery functionality."""
        tools_dir = Path(__file__).parent.parent / "ai_infrastructure" / "mcp" / "tools"
        
        if tools_dir.exists():
            discovered_tools = tool_registry.discover_tools()
            assert isinstance(discovered_tools, dict)
            
            # Should find at least the main categories
            expected_categories = [
                'file_tools', 'conversion_tools', 'image_tools',
                'text_tools', 'web_tools', 'system_tools'
            ]
            
            for category in expected_categories:
                if category in discovered_tools:
                    assert isinstance(discovered_tools[category], list)
    
    def test_tool_registration(self, tool_registry):
        """Test tool registration functionality."""
        # Create a mock tool
        mock_tool = {
            'name': 'test_tool',
            'category': 'test_tools',
            'module': 'test_module',
            'class_name': 'TestTool',
            'functions': ['execute'],
            'description': 'A test tool'
        }
        
        # Register the tool
        success = tool_registry.register_tool(mock_tool)
        assert success is True
        
        # Verify registration
        tool_key = f"{mock_tool['category']}.{mock_tool['name']}"
        assert tool_key in tool_registry.tools
    
    def test_mcp_server_initialization(self, mcp_server):
        """Test MCPServer initialization."""
        assert mcp_server is not None
        assert hasattr(mcp_server, 'tool_registry')
        assert hasattr(mcp_server, 'server')
    
    @pytest.mark.asyncio
    async def test_tool_execution(self, mcp_server):
        """Test tool execution through MCP server."""
        # This would test actual tool execution
        # For now, test the method exists and can be called
        assert hasattr(mcp_server, 'execute_tool')
        
        # Test with non-existent tool (should handle gracefully)
        result = await mcp_server.execute_tool("nonexistent.tool", {})
        assert 'error' in result


class TestIntegration:
    """Integration tests for AI infrastructure components."""
    
    def test_rag_with_ai_integration(self):
        """Test RAG system integration with AI models."""
        # This would test the full pipeline from document ingestion
        # to context retrieval to AI response generation
        pass
    
    def test_mcp_tool_chaining(self):
        """Test chaining multiple MCP tools together."""
        # Test executing a sequence of tools
        # e.g., file scan -> document conversion -> text analysis
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])