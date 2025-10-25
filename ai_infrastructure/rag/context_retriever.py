"""
Context retriever: query the vector DB and return assembled context for prompting a language model.
"""
from typing import List, Optional, Dict, Any
import tiktoken  # For more accurate token counting
from .chromadb_manager import ChromaDBManager

class ContextRetriever:
    def __init__(
        self, 
        collection_name: str = "default", 
        persist_directory: str = "./chroma_db", 
        embedding_model_name: str = "all-MiniLM-L6-v2",
        token_encoding: str = "cl100k_base"  # OpenAI's tokenizer
    ):
        self.collection_name = collection_name
        self.mgr = ChromaDBManager(
            persist_directory=persist_directory, 
            embedding_model_name=embedding_model_name
        )
        try:
            self.encoding = tiktoken.get_encoding(token_encoding)
        except:
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """Accurately count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback to character count approximation
            return len(text) // 4  # Rough approximation
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[dict]:
        """Retrieve documents with optional filtering."""
        return self.mgr.query(
            self.collection_name, 
            query, 
            top_k,
            filter_conditions=filter_conditions
        )
    
    def assemble_context(
        self, 
        query: str, 
        top_k: int = 5, 
        max_tokens: int = 1500,
        include_sources: bool = True,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> str:
        """Assemble context with token limit management."""
        hits = self.retrieve(query, top_k, filter_conditions)
        
        if not hits:
            return "No relevant context found."
        
        pieces = []
        total_tokens = 0
        
        for h in hits:
            doc = h.get("document", "").strip()
            meta = h.get("metadata", {})
            
            if include_sources:
                src = meta.get("source", "unknown")
                chunk_id = meta.get("chunk", "n/a")
                similarity = h.get("similarity", 0)
                
                # Format with source and similarity score
                piece = f"[Source: {src} | Chunk: {chunk_id} | Score: {similarity:.3f}]\n{doc}\n\n"
            else:
                piece = f"{doc}\n\n"
            
            piece_tokens = self.count_tokens(piece)
            
            # Check if adding this piece would exceed the limit
            if total_tokens + piece_tokens > max_tokens:
                # Try to add a truncated version if we have space
                if total_tokens < max_tokens:
                    available_tokens = max_tokens - total_tokens
                    if available_tokens > 50:  # Only truncate if we have meaningful space
                        truncated_doc = self._truncate_to_tokens(doc, available_tokens - 20)
                        if include_sources:
                            piece = f"[Source: {src} | Chunk: {chunk_id} | Score: {similarity:.3f} | TRUNCATED]\n{truncated_doc}...\n\n"
                        else:
                            piece = f"{truncated_doc}...\n\n"
                        pieces.append(piece)
                break
            
            pieces.append(piece)
            total_tokens += piece_tokens
        
        context = "".join(pieces).strip()
        
        # Add summary of context statistics
        if include_sources and len(hits) > 0:
            context += f"\n\n[Retrieved {len(pieces)} of {len(hits)} relevant documents | Total tokens: {total_tokens}]"
        
        return context
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens."""
        if self.encoding:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            return self.encoding.decode(truncated_tokens)
        else:
            # Character-based fallback
            max_chars = max_tokens * 4
            return text[:max_chars] + "..." if len(text) > max_chars else text
    
    def retrieve_and_format_prompt(
        self, 
        user_query: str, 
        instruction: Optional[str] = None, 
        top_k: int = 5,
        max_context_tokens: int = 1500,
        filter_conditions: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None
    ) -> str:
        """Retrieve context and format into a complete prompt."""
        instruction = instruction or "Use the provided context to answer the user's question. If the answer is not contained in the context, say you don't know."
        
        context = self.assemble_context(
            user_query, 
            top_k, 
            max_tokens=max_context_tokens,
            filter_conditions=filter_conditions
        )
        
        if prompt_template:
            return prompt_template.format(
                instruction=instruction,
                context=context,
                question=user_query
            )
        
        return f"{instruction}\n\nContext:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"
    
    def get_retrieval_metrics(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Get metrics about the retrieval process."""
        hits = self.retrieve(query, top_k)
        
        return {
            "total_hits": len(hits),
            "sources": list(set(h.get("metadata", {}).get("source", "unknown") for h in hits)),
            "similarity_scores": [h.get("similarity", 0) for h in hits],
            "average_similarity": sum(h.get("similarity", 0) for h in hits) / len(hits) if hits else 0
        }