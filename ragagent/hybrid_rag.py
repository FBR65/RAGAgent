"""
Hybrid RAG System combining traditional vector search with agentic navigation
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
import httpx

from .pipeline import AgenticRAGPipeline, ProcessingRequest, ProcessingResponse
from .models import Chunk, AnswerWithCitations, VerificationResult
from .utils import CacheManager, MemoryCache, cache_results
from .config import get_config
from .ollama_client import OllamaEmbeddingClient

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategy selection"""

    VECTOR_ONLY = "vector_only"
    AGENT_ONLY = "agent_only"
    HYBRID = "hybrid"
    HYBRID_WEIGHTED = "hybrid_weighted"


@dataclass
class VectorSearchResult:
    """Result from vector search"""

    chunks: List[Chunk]
    scores: List[float]
    search_time: float
    strategy: str


@dataclass
class HybridSearchResult:
    """Result from hybrid search"""

    vector_results: VectorSearchResult
    agent_results: List[Chunk]
    combined_results: List[Chunk]
    combination_strategy: str
    search_time: float


class BGEEmbedder:
    """BGE-M3 embedding client using Ollama"""

    def __init__(
        self,
        model_name: str = "bge-m3:latest",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.client = OllamaEmbeddingClient(model_name, base_url)
        self.logger = logging.getLogger(self.__class__.__name__)

    def embed(self, text: str) -> List[float]:
        """Embed a single text"""
        return self.client.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        return self.client.embed_batch(texts)


class QdrantVectorDB:
    """Qdrant vector database client"""

    def __init__(
        self,
        collection_name: str = "ragagent_hybrid",
        url: str = "http://localhost:6333",
        api_key: str = None,
        embedder: BGEEmbedder = None,
    ):
        self.collection_name = collection_name
        self.url = url
        self.api_key = api_key
        self.embedder = embedder or BGEEmbedder()

        try:
            self.client = QdrantClient(url=url, api_key=api_key)
            self.logger = logging.getLogger(self.__class__.__name__)
            self._ensure_collection()
        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def _ensure_collection(self):
        """Ensure the collection exists"""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                self.logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest.VectorParams(
                        size=1024,  # BGE-M3 dimension
                        distance=rest.Distance.COSINE,
                    ),
                )
            else:
                self.logger.info(f"Using existing collection: {self.collection_name}")

        except Exception as e:
            self.logger.error(f"Error ensuring collection: {e}")
            raise

    def index_chunks(self, chunks: List[Chunk], batch_size: int = 100):
        """Index chunks in Qdrant"""
        self.logger.info(f"Indexing {len(chunks)} chunks in Qdrant")

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_points = []

            for chunk in batch:
                # Generate embedding
                embedding = self.embedder.embed(chunk.text)

                # Create point
                point = rest.PointStruct(
                    id=chunk.id,
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "token_count": chunk.token_count,
                        "metadata": getattr(chunk, "metadata", {}),
                    },
                )
                batch_points.append(point)

            # Upload batch
            try:
                self.client.upload_points(
                    collection_name=self.collection_name, points=batch_points
                )
                self.logger.info(
                    f"Indexed batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}"
                )
            except Exception as e:
                self.logger.error(f"Error indexing batch {i // batch_size + 1}: {e}")
                raise

        self.logger.info(f"Successfully indexed {len(chunks)} chunks")

    def search(
        self, query: str, limit: int = 10, score_threshold: float = 0.1
    ) -> VectorSearchResult:
        """Perform vector search"""
        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = self.embedder.embed(query)

            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            # Convert to chunks
            chunks = []
            scores = []

            for point in search_result:
                # Create chunk from search result
                chunk = Chunk(
                    id=point.id,
                    text=point.payload["text"],
                    token_count=point.payload["token_count"],
                )
                chunks.append(chunk)
                scores.append(point.score)

            search_time = time.time() - start_time

            self.logger.info(
                f"Vector search completed in {search_time:.3f}s, found {len(chunks)} chunks"
            )

            return VectorSearchResult(
                chunks=chunks,
                scores=scores,
                search_time=search_time,
                strategy="qdrant_bge",
            )

        except Exception as e:
            self.logger.error(f"Error during vector search: {e}")
            raise

    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )
            return {
                "name": collection_info.name,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status,
                "config": collection_info.config.dict(),
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            raise


class TraditionalVectorSearch:
    """Traditional vector-based search using BGE-M3 and Qdrant"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.vector_db = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def initialize_vector_db(self, embedder: BGEEmbedder = None):
        """Initialize vector database"""
        self.vector_db = QdrantVectorDB(
            collection_name=self.config.get("collection_name", "ragagent_hybrid"),
            url=self.config.get("url", "http://localhost:6333"),
            api_key=self.config.get("api_key"),
            embedder=embedder or BGEEmbedder(),
        )

    def index_chunks(self, chunks: List[Chunk]):
        """Index chunks for vector search"""
        if self.vector_db is None:
            self.initialize_vector_db()

        self.vector_db.index_chunks(chunks)

    def search(self, query: str, top_k: int = 10) -> VectorSearchResult:
        """Perform vector search"""
        if self.vector_db is None:
            raise ValueError(
                "Vector database not initialized. Call initialize_vector_db() first."
            )

        return self.vector_db.search(query, limit=top_k)

    def get_relevant_chunks(
        self, query: str, threshold: float = 0.1, max_chunks: int = 20
    ) -> List[Chunk]:
        """Get chunks above similarity threshold"""
        results = self.search(query, top_k=max_chunks)

        # Filter by threshold
        filtered_chunks = [
            chunk
            for chunk, score in zip(results.chunks, results.scores)
            if score >= threshold
        ]

        self.logger.info(
            f"Filtered to {len(filtered_chunks)} chunks above threshold {threshold}"
        )
        return filtered_chunks


class HybridRAGSystem:
    """Hybrid RAG system combining traditional and agentic approaches"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.vector_search = TraditionalVectorSearch(
            self.config.get("vector_search", {})
        )
        self.agentic_pipeline = AgenticRAGPipeline()
        self.cache = CacheManager()

        # Hybrid configuration
        self.strategy = SearchStrategy.HYBRID
        self.vector_weight = self.config.get("hybrid_weights", {}).get(
            "vector_weight", 0.7
        )
        self.agent_weight = self.config.get("hybrid_weights", {}).get(
            "agent_weight", 0.3
        )
        self.similarity_threshold = self.config.get("similarity_threshold", 0.1)
        self.max_vector_chunks = self.config.get("max_vector_chunks", 10)
        self.max_agent_chunks = self.config.get("max_agent_chunks", 5)
        self.enable_caching = self.config.get("enable_caching", True)

        self.logger = logging.getLogger(self.__class__.__name__)

    def initialize_vector_db(self):
        """Initialize vector database"""
        self.vector_search.initialize_vector_db()

    def index_document(self, chunks: List[Chunk]):
        """Index document chunks in vector database"""
        self.logger.info(f"Indexing document with {len(chunks)} chunks")
        self.vector_search.index_chunks(chunks)

    @cache_results()
    def hybrid_search(self, query: str) -> HybridSearchResult:
        """Perform hybrid search combining vector and agentic approaches"""
        start_time = time.time()

        # Step 1: Vector search for fast initial filtering
        self.logger.info("Step 1: Performing vector search")
        vector_results = self.vector_search.search(query, top_k=self.max_vector_chunks)

        # Step 2: Agentic navigation on vector results
        self.logger.info("Step 2: Performing agentic navigation")
        agent_results = []

        if vector_results.chunks:
            try:
                # Use agentic pipeline to navigate vector results
                request = ProcessingRequest(
                    question=query,
                    document_path="",  # Not needed for chunk navigation
                    document_type=None,
                )

                # Override chunks with vector results
                self.agentic_pipeline.config.document_processing.max_initial_chunks = (
                    len(vector_results.chunks)
                )
                relevant_chunks = self.agentic_pipeline._navigate_to_relevant_chunks(
                    query, vector_results.chunks
                )
                agent_results = relevant_chunks[: self.max_agent_chunks]

            except Exception as e:
                self.logger.error(f"Error during agentic navigation: {e}")
                agent_results = vector_results.chunks[: self.max_agent_chunks]

        # Step 3: Combine results
        self.logger.info("Step 3: Combining search results")
        combined_results = self._combine_results(
            vector_results.chunks, agent_results, vector_results.scores
        )

        search_time = time.time() - start_time

        self.logger.info(f"Hybrid search completed in {search_time:.3f}s")

        return HybridSearchResult(
            vector_results=vector_results,
            agent_results=agent_results,
            combined_results=combined_results,
            combination_strategy="weighted_hybrid",
            search_time=search_time,
        )

    def _combine_results(
        self,
        vector_chunks: List[Chunk],
        agent_chunks: List[Chunk],
        vector_scores: List[float],
    ) -> List[Chunk]:
        """Combine vector and agent results using weighted scoring"""

        # Create lookup for agent chunks (assume perfect score)
        agent_chunk_ids = {chunk.id for chunk in agent_chunks}

        # Combine chunks with scores
        combined_chunks = []

        # Add vector chunks with their scores
        for chunk, score in zip(vector_chunks, vector_scores):
            if chunk.id not in agent_chunk_ids:  # Avoid duplicates
                combined_chunks.append((chunk, score * self.vector_weight))

        # Add agent chunks with weighted score
        for chunk in agent_chunks:
            if chunk.id not in [c[0].id for c in combined_chunks]:
                combined_chunks.append((chunk, self.agent_weight))

        # Sort by combined score
        combined_chunks.sort(key=lambda x: x[1], reverse=True)

        # Return chunks only
        return [chunk for chunk, score in combined_chunks]

    def process_query(self, query: str, use_hybrid: bool = True) -> Dict[str, Any]:
        """Process a query using hybrid search"""

        # Check cache first
        cache_key = f"hybrid_query:{hash(query)}"
        if self.enable_caching:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.logger.info("Returning cached result")
                return cached_result

        try:
            # Perform hybrid search
            if use_hybrid:
                search_result = self.hybrid_search(query)
                relevant_chunks = search_result.combined_results
            else:
                # Use only vector search
                search_result = self.vector_search.search(query)
                relevant_chunks = search_result.chunks

            # Generate answer using agentic pipeline
            self.logger.info("Generating answer from combined results")
            answer = self.agentic_pipeline._generate_answer(query, relevant_chunks)

            # Verify answer
            verification = None
            if self.agentic_pipeline.config.enable_verification:
                verification = self.agentic_pipeline._verify_answer(
                    query, answer, relevant_chunks
                )

            # Build result
            result = {
                "question": query,
                "answer": answer.answer,
                "citations": answer.citations,
                "confidence_score": answer.confidence_score,
                "verification": verification.dict() if verification else None,
                "search_strategy": "hybrid",
                "vector_search_time": search_result.vector_results.search_time,
                "total_search_time": search_result.search_time,
                "chunks_from_vector": len(search_result.vector_results.chunks),
                "chunks_from_agent": len(search_result.agent_results),
                "final_chunks_used": len(relevant_chunks),
                "metadata": {
                    "vector_weight": self.vector_weight,
                    "agent_weight": self.agent_weight,
                    "similarity_threshold": self.similarity_threshold,
                },
            }

            # Cache result
            if self.enable_caching:
                self.cache.set(cache_key, result, ttl=3600)  # 1 hour TTL

            return result

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get vector database collection information"""
        if self.vector_search.vector_db:
            return self.vector_search.vector_db.get_collection_info()
        else:
            raise ValueError("Vector database not initialized")

    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.logger.info("Cache cleared")

    def update_weights(self, vector_weight: float, agent_weight: float):
        """Update hybrid combination weights"""
        self.vector_weight = vector_weight
        self.agent_weight = agent_weight
        self.logger.info(
            f"Updated weights - vector: {vector_weight}, agent: {agent_weight}"
        )

    def set_strategy(self, strategy: SearchStrategy):
        """Set search strategy"""
        self.strategy = strategy
        self.logger.info(f"Set search strategy to: {strategy.value}")
