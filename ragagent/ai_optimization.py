"""
AI-powered optimization of chunking and navigation strategies
"""

import logging
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import redis
from enum import Enum

from .models import Chunk, ProcessingRequest, ProcessingResponse, DocumentType
from .config import get_config
from .utils import CacheManager

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies"""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SENTENCE_BOUNDARY = "sentence_boundary"
    PARAGRAPH = "paragraph"
    OVERLAPPING = "overlapping"
    ADAPTIVE = "adaptive"


class NavigationStrategy(Enum):
    """Available navigation strategies"""

    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    BEST_FIRST = "best_first"
    BEAM_SEARCH = "beam_search"
    MONTE_CARLO = "monte_carlo"
    HYBRID = "hybrid"


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies"""

    strategy: ChunkingStrategy
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    overlap_tokens: int = 100
    max_initial_chunks: int = 20
    semantic_similarity_threshold: float = 0.7
    adaptive_threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkingConfig":
        """Create from dictionary"""
        data["strategy"] = ChunkingStrategy(data["strategy"])
        return cls(**data)


@dataclass
class NavigationConfig:
    """Configuration for navigation strategies"""

    strategy: NavigationStrategy
    max_depth: int = 3
    beam_width: int = 3
    exploration_factor: float = 0.3
    confidence_threshold: float = 0.7
    max_iterations: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NavigationConfig":
        """Create from dictionary"""
        data["strategy"] = NavigationStrategy(data["strategy"])
        return cls(**data)


@dataclass
class OptimizationResult:
    """Result of optimization process"""

    strategy_name: str
    performance_metrics: Dict[str, float]
    execution_time: float
    memory_usage: float
    accuracy: float
    efficiency: float
    scalability: float
    robustness: float
    overall_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class DocumentAnalyzer:
    """Analyzes document characteristics for optimization"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Analysis parameters
        self.min_document_length = self.config.get("min_document_length", 1000)
        self.max_analysis_tokens = self.config.get("max_analysis_tokens", 5000)

        # Initialize ML components
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )

        self.logger.info("Document analyzer initialized")

    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Analyze document characteristics"""
        try:
            # Basic statistics
            doc_stats = self._calculate_basic_stats(text)

            # Text complexity analysis
            complexity = self._analyze_complexity(text)

            # Semantic structure analysis
            structure = self._analyze_structure(text)

            # Content type detection
            content_type = self._detect_content_type(text)

            # Language and style analysis
            language_style = self._analyze_language_style(text)

            # Topic analysis
            topics = self._extract_topics(text)

            analysis = {
                "basic_stats": doc_stats,
                "complexity": complexity,
                "structure": structure,
                "content_type": content_type,
                "language_style": language_style,
                "topics": topics,
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(f"Document analysis completed: {doc_stats}")
            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing document: {e}")
            return {}

    def _calculate_basic_stats(self, text: str) -> Dict[str, Any]:
        """Calculate basic document statistics"""
        words = text.split()
        sentences = text.split(".")
        paragraphs = text.split("\n\n")

        return {
            "total_characters": len(text),
            "total_words": len(words),
            "total_sentences": len(sentences),
            "total_paragraphs": len(paragraphs),
            "average_word_length": np.mean([len(word) for word in words])
            if words
            else 0,
            "average_sentence_length": len(words) / len(sentences) if sentences else 0,
            "average_paragraph_length": len(words) / len(paragraphs)
            if paragraphs
            else 0,
            "vocabulary_richness": len(set(words)) / len(words) if words else 0,
        }

    def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity"""
        words = text.split()
        sentences = text.split(".")

        # Calculate various complexity metrics
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0

        # Lexical diversity
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words) if words else 0

        # Syntactic complexity (simple approximation)
        long_words = [word for word in words if len(word) > 6]
        long_word_ratio = len(long_words) / len(words) if words else 0

        return {
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "lexical_diversity": lexical_diversity,
            "long_word_ratio": long_word_ratio,
            "complexity_score": (
                avg_word_length * 0.3
                + avg_sentence_length * 0.4
                + lexical_diversity * 0.2
                + long_word_ratio * 0.1
            ),
        }

    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure"""
        paragraphs = text.split("\n\n")
        sentences = text.split(".")

        # Paragraph length distribution
        para_lengths = [len(p.split()) for p in paragraphs if p.strip()]

        # Sentence length distribution
        sent_lengths = [len(s.split()) for s in sentences if s.strip()]

        # Structure patterns
        has_headings = any(len(p.split()) <= 5 and p.isupper() for p in paragraphs)
        has_lists = any("-" in p or "*" in p or "•" in p for p in paragraphs)
        has_tables = any("|" in p for p in paragraphs)

        return {
            "paragraph_count": len(paragraphs),
            "average_paragraph_length": np.mean(para_lengths) if para_lengths else 0,
            "paragraph_length_std": np.std(para_lengths) if para_lengths else 0,
            "sentence_count": len(sentences),
            "average_sentence_length": np.mean(sent_lengths) if sent_lengths else 0,
            "sentence_length_std": np.std(sent_lengths) if sent_lengths else 0,
            "has_headings": has_headings,
            "has_lists": has_lists,
            "has_tables": has_tables,
            "structure_type": self._classify_structure(paragraphs),
        }

    def _classify_structure(self, paragraphs: List[str]) -> str:
        """Classify document structure type"""
        # Simple heuristic-based classification
        if len(paragraphs) < 5:
            return "simple"
        elif len(paragraphs) < 20:
            return "structured"
        else:
            return "complex"

    def _detect_content_type(self, text: str) -> Dict[str, Any]:
        """Detect document content type"""
        # Simple keyword-based detection
        legal_keywords = ["contract", "agreement", "clause", "party", "liability"]
        technical_keywords = [
            "api",
            "endpoint",
            "request",
            "response",
            "authentication",
        ]
        medical_keywords = ["patient", "treatment", "diagnosis", "symptom", "therapy"]
        financial_keywords = ["payment", "invoice", "revenue", "expense", "profit"]

        text_lower = text.lower()

        keyword_counts = {
            "legal": sum(1 for kw in legal_keywords if kw in text_lower),
            "technical": sum(1 for kw in technical_keywords if kw in text_lower),
            "medical": sum(1 for kw in medical_keywords if kw in text_lower),
            "financial": sum(1 for kw in financial_keywords if kw in text_lower),
        }

        # Determine dominant content type
        dominant_type = max(keyword_counts, key=keyword_counts.get)
        confidence = (
            keyword_counts[dominant_type] / sum(keyword_counts.values())
            if sum(keyword_counts.values()) > 0
            else 0
        )

        return {
            "detected_type": dominant_type,
            "confidence": confidence,
            "keyword_counts": keyword_counts,
        }

    def _analyze_language_style(self, text: str) -> Dict[str, Any]:
        """Analyze language and style"""
        words = text.split()

        # Formality indicators
        formal_words = [
            "therefore",
            "however",
            "furthermore",
            "nevertheless",
            "consequently",
        ]
        informal_words = ["like", "you know", "actually", "basically", "literally"]

        text_lower = text.lower()

        formality_score = (
            (
                sum(1 for fw in formal_words if fw in text_lower)
                - sum(1 for iw in informal_words if iw in text_lower)
            )
            / len(words)
            if words
            else 0
        )

        # Readability indicators
        long_words = [word for word in words if len(word) > 6]
        long_word_ratio = len(long_words) / len(words) if words else 0

        return {
            "formality_score": formality_score,
            "long_word_ratio": long_word_ratio,
            "style_type": "formal"
            if formality_score > 0.1
            else "informal"
            if formality_score < -0.1
            else "neutral",
        }

    def _extract_topics(self, text: str) -> List[Dict[str, Any]]:
        """Extract key topics from text"""
        try:
            # Simple topic extraction using TF-IDF
            sentences = text.split(".")

            if len(sentences) < 3:
                return []

            # Vectorize sentences
            tfidf_matrix = self.vectorizer.fit_transform(sentences)

            # Extract top keywords
            feature_names = self.vectorizer.get_feature_names_out()
            topic_scores = tfidf_matrix.sum(axis=0).A1

            # Get top topics
            top_indices = np.argsort(topic_scores)[-5:][::-1]
            top_topics = [
                {
                    "keyword": feature_names[i],
                    "score": float(topic_scores[i]),
                    "sentences": [
                        s.strip() for s in sentences if feature_names[i] in s.lower()
                    ],
                }
                for i in top_indices
                if topic_scores[i] > 0
            ]

            return top_topics

        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}")
            return []


class ChunkingOptimizer:
    """Optimizes chunking strategies based on document characteristics"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.analyzer = DocumentAnalyzer(config)
        self.cache = CacheManager()

        # Optimization parameters
        self.optimization_timeout = self.config.get("optimization_timeout", 300)
        self.max_chunk_variants = self.config.get("max_chunk_variants", 10)

        # Strategy configurations
        self.strategies = {
            ChunkingStrategy.FIXED_SIZE: self._fixed_size_chunking,
            ChunkingStrategy.SEMANTIC: self._semantic_chunking,
            ChunkingStrategy.SENTENCE_BOUNDARY: self._sentence_boundary_chunking,
            ChunkingStrategy.PARAGRAPH: self._paragraph_chunking,
            ChunkingStrategy.OVERLAPPING: self._overlapping_chunking,
            ChunkingStrategy.ADAPTIVE: self._adaptive_chunking,
        }

        self.logger.info("Chunking optimizer initialized")

    def optimize_chunking(self, text: str, question: str = None) -> OptimizationResult:
        """Optimize chunking strategy for given document"""
        start_time = time.time()

        try:
            # Analyze document
            analysis = self.analyzer.analyze_document(text)

            # Generate strategy variants
            variants = self._generate_strategy_variants(analysis, question)

            # Evaluate each variant
            results = []
            for variant in variants:
                result = self._evaluate_chunking_variant(text, variant, analysis)
                results.append(result)

            # Select best strategy
            best_result = max(results, key=lambda x: x.overall_score)

            execution_time = time.time() - start_time

            self.logger.info(
                f"Chunking optimization completed in {execution_time:.2f}s"
            )
            self.logger.info(
                f"Best strategy: {best_result.strategy_name} (score: {best_result.overall_score:.3f})"
            )

            return best_result

        except Exception as e:
            self.logger.error(f"Error optimizing chunking: {e}")
            return OptimizationResult(
                strategy_name="fallback",
                performance_metrics={},
                execution_time=time.time() - start_time,
                memory_usage=0,
                accuracy=0.5,
                efficiency=0.5,
                scalability=0.5,
                robustness=0.5,
                overall_score=0.5,
            )

    def _generate_strategy_variants(
        self, analysis: Dict[str, Any], question: str = None
    ) -> List[ChunkingConfig]:
        """Generate strategy variants based on document analysis"""
        variants = []

        # Base strategies
        base_strategies = [
            ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE, max_chunk_size=1000),
            ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE, max_chunk_size=2000),
            ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE, max_chunk_size=3000),
            ChunkingConfig(strategy=ChunkingStrategy.SENTENCE_BOUNDARY),
            ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH),
            ChunkingConfig(strategy=ChunkingStrategy.SEMANTIC),
            ChunkingConfig(strategy=ChunkingStrategy.ADAPTIVE),
        ]

        # Adapt strategies based on document characteristics
        doc_length = analysis.get("basic_stats", {}).get("total_words", 0)
        complexity = analysis.get("complexity", {}).get("complexity_score", 0)
        structure_type = analysis.get("structure", {}).get("structure_type", "simple")

        # Adjust parameters based on analysis
        if doc_length > 5000:  # Long document
            for variant in base_strategies:
                if variant.strategy == ChunkingStrategy.FIXED_SIZE:
                    variant.max_chunk_size = min(variant.max_chunk_size * 2, 5000)

        if complexity > 0.7:  # Complex document
            for variant in base_strategies:
                if variant.strategy in [
                    ChunkingStrategy.SEMANTIC,
                    ChunkingStrategy.ADAPTIVE,
                ]:
                    variant.semantic_similarity_threshold = 0.5

        if structure_type == "complex":  # Complex structure
            for variant in base_strategies:
                if variant.strategy == ChunkingStrategy.OVERLAPPING:
                    variant.overlap_tokens = 200

        variants.extend(base_strategies)

        # Add question-specific variants if question provided
        if question:
            question_variants = self._generate_question_specific_variants(
                question, analysis
            )
            variants.extend(question_variants)

        return variants[: self.max_chunk_variants]

    def _generate_question_specific_variants(
        self, question: str, analysis: Dict[str, Any]
    ) -> List[ChunkingConfig]:
        """Generate question-specific chunking variants"""
        variants = []

        # Analyze question type
        question_lower = question.lower()

        if any(word in question_lower for word in ["what", "which", "list"]):
            # Factual questions - prefer smaller chunks
            variants.append(
                ChunkingConfig(
                    strategy=ChunkingStrategy.FIXED_SIZE,
                    max_chunk_size=1000,
                    overlap_tokens=50,
                )
            )

        elif any(word in question_lower for word in ["why", "how", "explain"]):
            # Explanatory questions - prefer larger chunks
            variants.append(
                ChunkingConfig(
                    strategy=ChunkingStrategy.FIXED_SIZE,
                    max_chunk_size=3000,
                    overlap_tokens=200,
                )
            )

        elif any(
            word in question_lower for word in ["compare", "contrast", "difference"]
        ):
            # Comparative questions - prefer semantic chunking
            variants.append(
                ChunkingConfig(
                    strategy=ChunkingStrategy.SEMANTIC,
                    semantic_similarity_threshold=0.6,
                )
            )

        return variants

    def _evaluate_chunking_variant(
        self, text: str, config: ChunkingConfig, analysis: Dict[str, Any]
    ) -> OptimizationResult:
        """Evaluate a chunking variant"""
        start_time = time.time()

        try:
            # Generate chunks
            chunks = self.strategies[config.strategy](text, config)

            # Calculate metrics
            metrics = self._calculate_chunk_metrics(chunks, analysis)

            # Evaluate performance
            performance = self._evaluate_chunk_performance(chunks, analysis)

            execution_time = time.time() - start_time

            # Calculate overall score
            overall_score = (
                performance["relevance"] * 0.3
                + performance["coherence"] * 0.2
                + performance["coverage"] * 0.2
                + metrics["efficiency"] * 0.15
                + metrics["consistency"] * 0.15
            )

            return OptimizationResult(
                strategy_name=config.strategy.value,
                performance_metrics=performance,
                execution_time=execution_time,
                memory_usage=metrics["memory_usage"],
                accuracy=performance["relevance"],
                efficiency=metrics["efficiency"],
                scalability=metrics["scalability"],
                robustness=metrics["consistency"],
                overall_score=overall_score,
            )

        except Exception as e:
            self.logger.error(f"Error evaluating chunking variant: {e}")
            return OptimizationResult(
                strategy_name=config.strategy.value,
                performance_metrics={},
                execution_time=time.time() - start_time,
                memory_usage=0,
                accuracy=0.1,
                efficiency=0.1,
                scalability=0.1,
                robustness=0.1,
                overall_score=0.1,
            )

    def _fixed_size_chunking(self, text: str, config: ChunkingConfig) -> List[Chunk]:
        """Fixed size chunking"""
        chunks = []
        words = text.split()
        chunk_size = config.max_chunk_size

        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i : i + chunk_size])
            chunks.append(
                Chunk(
                    id=f"fixed_{i // chunk_size}",
                    text=chunk_text,
                    token_count=len(chunk_text.split()),
                    document_type=DocumentType.TXT,
                )
            )

        return chunks

    def _semantic_chunking(self, text: str, config: ChunkingConfig) -> List[Chunk]:
        """Semantic chunking using similarity"""
        sentences = text.split(".")
        chunks = []

        if len(sentences) < 2:
            return [
                Chunk(
                    id="semantic_0",
                    text=text,
                    token_count=len(text.split()),
                    document_type=DocumentType.TXT,
                )
            ]

        # Vectorize sentences
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=100)
        try:
            sentence_vectors = vectorizer.fit_transform(
                [s.strip() for s in sentences if s.strip()]
            )

            # Cluster sentences
            n_clusters = min(len(sentences) // 3, 10)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(sentence_vectors)

                # Create chunks from clusters
                for cluster_id in range(n_clusters):
                    cluster_sentences = [
                        sentences[i]
                        for i, label in enumerate(cluster_labels)
                        if label == cluster_id
                    ]
                    chunk_text = " ".join(
                        [s.strip() for s in cluster_sentences if s.strip()]
                    )
                    if chunk_text:
                        chunks.append(
                            Chunk(
                                id=f"semantic_{cluster_id}",
                                text=chunk_text,
                                token_count=len(chunk_text.split()),
                                document_type=DocumentType.TXT,
                            )
                        )
            else:
                # Fallback to sentence boundary chunking
                return self._sentence_boundary_chunking(text, config)

        except Exception as e:
            self.logger.error(f"Error in semantic chunking: {e}")
            return self._sentence_boundary_chunking(text, config)

        return chunks

    def _sentence_boundary_chunking(
        self, text: str, config: ChunkingConfig
    ) -> List[Chunk]:
        """Sentence boundary chunking"""
        sentences = text.split(".")
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            if sentence.strip():
                sentence_tokens = len(sentence.split())

                if (
                    current_tokens + sentence_tokens > config.max_chunk_size
                    and current_chunk
                ):
                    chunks.append(
                        Chunk(
                            id=f"sentence_{len(chunks)}",
                            text=current_chunk.strip(),
                            token_count=current_tokens,
                            document_type=DocumentType.TXT,
                        )
                    )
                    current_chunk = sentence
                    current_tokens = sentence_tokens
                else:
                    current_chunk += " " + sentence
                    current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(
                Chunk(
                    id=f"sentence_{len(chunks)}",
                    text=current_chunk.strip(),
                    token_count=current_tokens,
                    document_type=DocumentType.TXT,
                )
            )

        return chunks

    def _paragraph_chunking(self, text: str, config: ChunkingConfig) -> List[Chunk]:
        """Paragraph chunking"""
        paragraphs = text.split("\n\n")
        chunks = []

        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                chunks.append(
                    Chunk(
                        id=f"paragraph_{i}",
                        text=paragraph.strip(),
                        token_count=len(paragraph.split()),
                        document_type=DocumentType.TXT,
                    )
                )

        return chunks

    def _overlapping_chunking(self, text: str, config: ChunkingConfig) -> List[Chunk]:
        """Overlapping chunking"""
        words = text.split()
        chunk_size = config.max_chunk_size
        overlap_size = config.overlap_tokens

        chunks = []

        for i in range(0, len(words), chunk_size - overlap_size):
            chunk_text = " ".join(words[i : i + chunk_size])
            chunks.append(
                Chunk(
                    id=f"overlap_{i}",
                    text=chunk_text,
                    token_count=len(chunk_text.split()),
                    document_type=DocumentType.TXT,
                )
            )

        return chunks

    def _adaptive_chunking(self, text: str, config: ChunkingConfig) -> List[Chunk]:
        """Adaptive chunking based on content"""
        # First analyze document structure
        analysis = self.analyzer.analyze_document(text)

        # Choose base strategy based on analysis
        structure_type = analysis.get("structure", {}).get("structure_type", "simple")

        if structure_type == "complex":
            return self._semantic_chunking(text, config)
        elif structure_type == "structured":
            return self._paragraph_chunking(text, config)
        else:
            return self._sentence_boundary_chunking(text, config)

    def _calculate_chunk_metrics(
        self, chunks: List[Chunk], analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate chunk quality metrics"""
        if not chunks:
            return {}

        # Basic metrics
        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_chunk_size = total_tokens / len(chunks)
        size_variance = np.var([chunk.token_count for chunk in chunks])

        # Efficiency metrics
        efficiency = (
            1.0 - (size_variance / (avg_chunk_size**2)) if avg_chunk_size > 0 else 0
        )

        # Scalability metrics
        scalability = min(1.0, len(chunks) / 20)  # Normalize to 20 chunks

        # Memory usage estimate
        memory_usage = total_tokens * 0.001  # Rough estimate in MB

        return {
            "efficiency": efficiency,
            "scalability": scalability,
            "consistency": 1.0 - (size_variance / 10000),  # Normalize variance
            "memory_usage": memory_usage,
        }

    def _evaluate_chunk_performance(
        self, chunks: List[Chunk], analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate chunk performance based on document characteristics"""
        if not chunks:
            return {}

        # Relevance based on document structure
        structure_type = analysis.get("structure", {}).get("structure_type", "simple")

        if structure_type == "complex" and len(chunks) > 10:
            relevance = 0.8  # Good for complex documents
        elif structure_type == "simple" and len(chunks) < 10:
            relevance = 0.9  # Good for simple documents
        else:
            relevance = 0.7

        # Coherence based on chunk size distribution
        chunk_sizes = [chunk.token_count for chunk in chunks]
        size_variance = np.var(chunk_sizes)
        coherence = 1.0 - min(1.0, size_variance / 10000)

        # Coverage based on document length
        doc_words = analysis.get("basic_stats", {}).get("total_words", 0)
        chunk_words = sum(chunk.token_count for chunk in chunks)
        coverage = min(1.0, chunk_words / doc_words) if doc_words > 0 else 0

        return {"relevance": relevance, "coherence": coherence, "coverage": coverage}


class NavigationOptimizer:
    """Optimizes navigation strategies based on document and query characteristics"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Optimization parameters
        self.max_navigation_variants = self.config.get("max_navigation_variants", 8)
        self.optimization_timeout = self.config.get("optimization_timeout", 300)

        # Strategy configurations
        self.strategies = {
            NavigationStrategy.DEPTH_FIRST: self._depth_first_navigation,
            NavigationStrategy.BREADTH_FIRST: self._breadth_first_navigation,
            NavigationStrategy.BEST_FIRST: self._best_first_navigation,
            NavigationStrategy.BEAM_SEARCH: self._beam_search_navigation,
            NavigationStrategy.MONTE_CARLO: self._monte_carlo_navigation,
            NavigationStrategy.HYBRID: self._hybrid_navigation,
        }

        self.logger.info("Navigation optimizer initialized")

    def optimize_navigation(
        self,
        chunks: List[Chunk],
        question: str,
        document_analysis: Dict[str, Any] = None,
    ) -> OptimizationResult:
        """Optimize navigation strategy for given chunks and question"""
        start_time = time.time()

        try:
            # Generate strategy variants
            variants = self._generate_navigation_variants(
                chunks, question, document_analysis
            )

            # Evaluate each variant
            results = []
            for variant in variants:
                result = self._evaluate_navigation_variant(chunks, question, variant)
                results.append(result)

            # Select best strategy
            best_result = max(results, key=lambda x: x.overall_score)

            execution_time = time.time() - start_time

            self.logger.info(
                f"Navigation optimization completed in {execution_time:.2f}s"
            )
            self.logger.info(
                f"Best strategy: {best_result.strategy_name} (score: {best_result.overall_score:.3f})"
            )

            return best_result

        except Exception as e:
            self.logger.error(f"Error optimizing navigation: {e}")
            return OptimizationResult(
                strategy_name="fallback",
                performance_metrics={},
                execution_time=time.time() - start_time,
                memory_usage=0,
                accuracy=0.5,
                efficiency=0.5,
                scalability=0.5,
                robustness=0.5,
                overall_score=0.5,
            )

    def _generate_navigation_variants(
        self,
        chunks: List[Chunk],
        question: str,
        document_analysis: Dict[str, Any] = None,
    ) -> List[NavigationConfig]:
        """Generate navigation strategy variants"""
        variants = []

        # Base strategies
        base_strategies = [
            NavigationConfig(strategy=NavigationStrategy.DEPTH_FIRST, max_depth=3),
            NavigationConfig(strategy=NavigationStrategy.BREADTH_FIRST, max_depth=3),
            NavigationConfig(
                strategy=NavigationStrategy.BEST_FIRST, confidence_threshold=0.7
            ),
            NavigationConfig(strategy=NavigationStrategy.BEAM_SEARCH, beam_width=3),
            NavigationConfig(
                strategy=NavigationStrategy.MONTE_CARLO, exploration_factor=0.3
            ),
            NavigationConfig(strategy=NavigationStrategy.HYBRID, max_depth=2),
        ]

        # Adapt strategies based on document characteristics
        if document_analysis:
            chunk_count = len(chunks)
            complexity = document_analysis.get("complexity", {}).get(
                "complexity_score", 0
            )
            structure_type = document_analysis.get("structure", {}).get(
                "structure_type", "simple"
            )

            # Adjust parameters based on analysis
            if chunk_count > 50:  # Large document
                for variant in base_strategies:
                    if variant.strategy in [
                        NavigationStrategy.DEPTH_FIRST,
                        NavigationStrategy.BREADTH_FIRST,
                    ]:
                        variant.max_depth = min(variant.max_depth + 1, 5)

            if complexity > 0.7:  # Complex document
                for variant in base_strategies:
                    if variant.strategy == NavigationStrategy.BEST_FIRST:
                        variant.confidence_threshold = 0.5

        # Add question-specific variants
        question_variants = self._generate_question_specific_variants(
            question, len(chunks)
        )
        variants.extend(question_variants)

        variants.extend(base_strategies)
        return variants[: self.max_navigation_variants]

    def _generate_question_specific_variants(
        self, question: str, chunk_count: int
    ) -> List[NavigationConfig]:
        """Generate question-specific navigation variants"""
        variants = []
        question_lower = question.lower()

        if any(word in question_lower for word in ["what", "which", "list"]):
            # Factual questions - prefer breadth-first
            variants.append(
                NavigationConfig(strategy=NavigationStrategy.BREADTH_FIRST, max_depth=2)
            )

        elif any(word in question_lower for word in ["why", "how", "explain"]):
            # Explanatory questions - prefer depth-first
            variants.append(
                NavigationConfig(strategy=NavigationStrategy.DEPTH_FIRST, max_depth=4)
            )

        elif any(
            word in question_lower for word in ["compare", "contrast", "difference"]
        ):
            # Comparative questions - prefer beam search
            variants.append(
                NavigationConfig(strategy=NavigationStrategy.BEAM_SEARCH, beam_width=5)
            )

        return variants

    def _evaluate_navigation_variant(
        self, chunks: List[Chunk], question: str, config: NavigationConfig
    ) -> OptimizationResult:
        """Evaluate a navigation variant"""
        start_time = time.time()

        try:
            # Simulate navigation
            navigation_result = self.strategies[config.strategy](
                chunks, question, config
            )

            # Calculate metrics
            metrics = self._calculate_navigation_metrics(navigation_result, chunks)

            execution_time = time.time() - start_time

            # Calculate overall score
            overall_score = (
                metrics["accuracy"] * 0.4
                + metrics["efficiency"] * 0.3
                + metrics["completeness"] * 0.2
                + metrics["consistency"] * 0.1
            )

            return OptimizationResult(
                strategy_name=config.strategy.value,
                performance_metrics=metrics,
                execution_time=execution_time,
                memory_usage=metrics["memory_usage"],
                accuracy=metrics["accuracy"],
                efficiency=metrics["efficiency"],
                scalability=metrics["scalability"],
                robustness=metrics["consistency"],
                overall_score=overall_score,
            )

        except Exception as e:
            self.logger.error(f"Error evaluating navigation variant: {e}")
            return OptimizationResult(
                strategy_name=config.strategy.value,
                performance_metrics={},
                execution_time=time.time() - start_time,
                memory_usage=0,
                accuracy=0.1,
                efficiency=0.1,
                scalability=0.1,
                robustness=0.1,
                overall_score=0.1,
            )

    def _depth_first_navigation(
        self, chunks: List[Chunk], question: str, config: NavigationConfig
    ) -> Dict[str, Any]:
        """Depth-first navigation simulation"""
        selected_chunks = []
        visited = set()

        def dfs(chunk_id, depth):
            if depth >= config.max_depth or chunk_id in visited:
                return

            visited.add(chunk_id)
            selected_chunks.append(chunk_id)

            # Find related chunks (simplified)
            related_chunks = self._find_related_chunks(chunks, chunk_id, question)
            for related_chunk in related_chunks[:2]:  # Limit branching
                dfs(related_chunk["id"], depth + 1)

        # Start from most relevant chunk
        start_chunk = self._find_most_relevant_chunk(chunks, question)
        if start_chunk:
            dfs(start_chunk["id"], 0)

        return {
            "selected_chunks": selected_chunks,
            "visited_chunks": list(visited),
            "depth": config.max_depth,
            "strategy": "depth_first",
        }

    def _breadth_first_navigation(
        self, chunks: List[Chunk], question: str, config: NavigationConfig
    ) -> Dict[str, Any]:
        """Breadth-first navigation simulation"""
        from collections import deque

        selected_chunks = []
        visited = set()
        queue = deque()

        # Start with most relevant chunks
        relevant_chunks = self._find_relevant_chunks(chunks, question, top_k=5)
        for chunk in relevant_chunks:
            queue.append(chunk["id"])
            visited.add(chunk["id"])
            selected_chunks.append(chunk["id"])

        # BFS traversal
        current_depth = 0
        while queue and current_depth < config.max_depth:
            level_size = len(queue)

            for _ in range(level_size):
                chunk_id = queue.popleft()

                # Find related chunks
                related_chunks = self._find_related_chunks(chunks, chunk_id, question)
                for related_chunk in related_chunks:
                    if related_chunk["id"] not in visited:
                        visited.add(related_chunk["id"])
                        selected_chunks.append(related_chunk["id"])
                        queue.append(related_chunk["id"])

            current_depth += 1

        return {
            "selected_chunks": selected_chunks,
            "visited_chunks": list(visited),
            "depth": current_depth,
            "strategy": "breadth_first",
        }

    def _best_first_navigation(
        self, chunks: List[Chunk], question: str, config: NavigationConfig
    ) -> Dict[str, Any]:
        """Best-first navigation simulation"""
        selected_chunks = []
        visited = set()

        # Priority queue (simplified)
        candidates = self._find_relevant_chunks(chunks, question, top_k=len(chunks))

        while candidates and len(selected_chunks) < 20:  # Limit total chunks
            # Select best candidate
            best_candidate = candidates.pop(0)
            candidate_id = best_candidate["id"]

            if candidate_id in visited:
                continue

            visited.add(candidate_id)
            selected_chunks.append(candidate_id)

            # Find related chunks and add to candidates
            related_chunks = self._find_related_chunks(chunks, candidate_id, question)
            for related_chunk in related_chunks:
                if related_chunk["id"] not in visited:
                    # Calculate combined score
                    related_chunk["score"] = (
                        related_chunk.get("score", 0) * config.confidence_threshold
                    )
                    candidates.append(related_chunk)

            # Re-sort candidates by score
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

        return {
            "selected_chunks": selected_chunks,
            "visited_chunks": list(visited),
            "strategy": "best_first",
        }

    def _beam_search_navigation(
        self, chunks: List[Chunk], question: str, config: NavigationConfig
    ) -> Dict[str, Any]:
        """Beam search navigation simulation"""
        selected_chunks = []
        visited = set()

        # Initial beam
        beam = self._find_relevant_chunks(chunks, question, top_k=config.beam_width)

        for _ in range(config.max_depth):
            new_beam = []

            for chunk in beam:
                chunk_id = chunk["id"]

                if chunk_id in visited:
                    continue

                visited.add(chunk_id)
                selected_chunks.append(chunk_id)

                # Find related chunks
                related_chunks = self._find_related_chunks(chunks, chunk_id, question)
                for related_chunk in related_chunks:
                    if related_chunk["id"] not in visited:
                        # Calculate score
                        score = chunk.get("score", 0) + related_chunk.get("score", 0)
                        new_beam.append(
                            {
                                "id": related_chunk["id"],
                                "score": score,
                                "parent": chunk_id,
                            }
                        )

            # Select top k for next beam
            new_beam.sort(key=lambda x: x["score"], reverse=True)
            beam = new_beam[: config.beam_width]

        return {
            "selected_chunks": selected_chunks,
            "visited_chunks": list(visited),
            "strategy": "beam_search",
        }

    def _monte_carlo_navigation(
        self, chunks: List[Chunk], question: str, config: NavigationConfig
    ) -> Dict[str, Any]:
        """Monte Carlo navigation simulation"""
        selected_chunks = []
        visited = set()

        # Multiple random walks
        for _ in range(config.max_iterations):
            current_chunk = self._find_most_relevant_chunk(chunks, question)
            if not current_chunk:
                continue

            walk = []
            current_id = current_chunk["id"]

            for _ in range(config.max_depth):
                if current_id in visited:
                    break

                visited.add(current_id)
                walk.append(current_id)

                # Random walk to related chunk
                related_chunks = self._find_related_chunks(chunks, current_id, question)
                if related_chunks:
                    # Select based on exploration factor
                    if np.random.random() < config.exploration_factor:
                        # Explore - random selection
                        next_chunk = np.random.choice(related_chunks)
                    else:
                        # Exploit - best selection
                        next_chunk = max(
                            related_chunks, key=lambda x: x.get("score", 0)
                        )

                    current_id = next_chunk["id"]
                else:
                    break

            selected_chunks.extend(walk)

        return {
            "selected_chunks": list(set(selected_chunks)),  # Remove duplicates
            "visited_chunks": list(visited),
            "strategy": "monte_carlo",
        }

    def _hybrid_navigation(
        self, chunks: List[Chunk], question: str, config: NavigationConfig
    ) -> Dict[str, Any]:
        """Hybrid navigation combining multiple strategies"""
        selected_chunks = []
        visited = set()

        # Phase 1: Best-first for initial selection
        best_chunks = self._best_first_navigation(chunks, question, config)
        selected_chunks.extend(best_chunks["selected_chunks"])
        visited.update(best_chunks["visited_chunks"])

        # Phase 2: Breadth-first for exploration
        remaining_chunks = [chunk for chunk in chunks if chunk.id not in visited]
        if remaining_chunks:
            breadth_config = NavigationConfig(
                strategy=NavigationStrategy.BREADTH_FIRST,
                max_depth=config.max_depth - 1,
            )
            breadth_chunks = self._breadth_first_navigation(
                remaining_chunks, question, breadth_config
            )
            selected_chunks.extend(breadth_chunks["selected_chunks"])
            visited.update(breadth_chunks["visited_chunks"])

        return {
            "selected_chunks": selected_chunks,
            "visited_chunks": list(visited),
            "strategy": "hybrid",
        }

    def _find_most_relevant_chunk(
        self, chunks: List[Chunk], question: str
    ) -> Optional[Dict[str, Any]]:
        """Find most relevant chunk for question"""
        relevant_chunks = self._find_relevant_chunks(chunks, question, top_k=1)
        return relevant_chunks[0] if relevant_chunks else None

    def _find_relevant_chunks(
        self, chunks: List[Chunk], question: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find relevant chunks for question"""
        # Simple relevance scoring based on keyword overlap
        question_words = set(question.lower().split())

        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk.text.lower().split())
            overlap = len(question_words.intersection(chunk_words))
            score = overlap / len(question_words) if question_words else 0

            scored_chunks.append({"id": chunk.id, "score": score, "chunk": chunk})

        # Sort by score and return top k
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def _find_related_chunks(
        self, chunks: List[Chunk], chunk_id: str, question: str
    ) -> List[Dict[str, Any]]:
        """Find chunks related to a given chunk"""
        # Simplified relatedness based on text similarity
        target_chunk = next((chunk for chunk in chunks if chunk.id == chunk_id), None)
        if not target_chunk:
            return []

        # Simple keyword overlap with target chunk
        target_words = set(target_chunk.text.lower().split())

        related_chunks = []
        for chunk in chunks:
            if chunk.id != chunk_id:
                chunk_words = set(chunk.text.lower().split())
                overlap = len(target_words.intersection(chunk_words))
                score = overlap / len(target_words) if target_words else 0

                if score > 0.1:  # Minimum threshold
                    related_chunks.append(
                        {"id": chunk.id, "score": score, "chunk": chunk}
                    )

        return related_chunks

    def _calculate_navigation_metrics(
        self, navigation_result: Dict[str, Any], chunks: List[Chunk]
    ) -> Dict[str, float]:
        """Calculate navigation performance metrics"""
        selected_count = len(navigation_result["selected_chunks"])
        visited_count = len(navigation_result["visited_chunks"])
        total_count = len(chunks)

        # Accuracy based on coverage
        coverage = selected_count / total_count if total_count > 0 else 0

        # Efficiency based on selected vs visited ratio
        efficiency = selected_count / visited_count if visited_count > 0 else 0

        # Completeness based on strategy performance
        completeness = min(1.0, selected_count / 10)  # Normalize to 10 chunks

        # Consistency based on strategy characteristics
        strategy = navigation_result.get("strategy", "unknown")
        if strategy in ["depth_first", "breadth_first"]:
            consistency = 0.8
        elif strategy in ["best_first", "beam_search"]:
            consistency = 0.9
        else:
            consistency = 0.7

        # Memory usage estimate
        memory_usage = visited_count * 0.001  # Rough estimate in MB

        return {
            "accuracy": coverage,
            "efficiency": efficiency,
            "completeness": completeness,
            "consistency": consistency,
            "memory_usage": memory_usage,
            "scalability": min(1.0, visited_count / 50),  # Normalize to 50 chunks
        }


class AIOptimizationManager:
    """Main manager for AI-powered optimization"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize optimizers
        self.chunking_optimizer = ChunkingOptimizer(config)
        self.navigation_optimizer = NavigationOptimizer(config)

        # Cache for optimization results
        self.cache = CacheManager()

        # Redis client for distributed caching
        self.redis = redis.Redis(
            host=self.config.get("redis_host", "localhost"),
            port=self.config.get("redis_port", 6379),
        )

        self.logger.info("AI optimization manager initialized")

    def optimize_processing(self, text: str, question: str = None) -> Dict[str, Any]:
        """Optimize both chunking and navigation strategies"""
        start_time = time.time()

        try:
            # Create cache key
            cache_key = f"optimization:{hash(text)}:{hash(question)}"

            # Check cache first
            cached_result = self.redis.get(cache_key)
            if cached_result:
                self.logger.info("Using cached optimization result")
                return json.loads(cached_result)

            # Analyze document
            document_analysis = self.chunking_optimizer.analyzer.analyze_document(text)

            # Optimize chunking
            chunking_result = self.chunking_optimizer.optimize_chunking(text, question)

            # Generate chunks using best strategy
            best_chunking_config = ChunkingConfig(
                strategy=ChunkingStrategy(chunking_result.strategy_name),
                **{
                    k: v
                    for k, v in chunking_result.performance_metrics.items()
                    if k in ["max_chunk_size", "min_chunk_size", "overlap_tokens"]
                },
            )

            chunks = self.chunking_optimizer.strategies[best_chunking_config.strategy](
                text, best_chunking_config
            )

            # Optimize navigation
            navigation_result = self.navigation_optimizer.optimize_navigation(
                chunks, question, document_analysis
            )

            # Combine results
            optimization_result = {
                "chunking_optimization": chunking_result.to_dict(),
                "navigation_optimization": navigation_result.to_dict(),
                "document_analysis": document_analysis,
                "recommended_chunking_config": best_chunking_config.to_dict(),
                "recommended_navigation_config": NavigationConfig(
                    strategy=NavigationStrategy(navigation_result.strategy_name),
                    **{
                        k: v
                        for k, v in navigation_result.performance_metrics.items()
                        if k in ["max_depth", "beam_width", "confidence_threshold"]
                    },
                ).to_dict(),
                "total_optimization_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

            # Convert enums to strings before JSON serialization
            def make_json_serializable(obj):
                """Convert enums and other non-serializable objects to strings"""
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, Enum):
                    return obj.value
                elif hasattr(obj, "__dict__"):
                    return make_json_serializable(obj.__dict__)
                else:
                    return obj

            serializable_result = make_json_serializable(optimization_result)

            # Cache result
            self.redis.setex(
                cache_key, 3600, json.dumps(serializable_result)
            )  # 1 hour TTL

            self.logger.info(
                f"Processing optimization completed in {optimization_result['total_optimization_time']:.2f}s"
            )
            return optimization_result

        except Exception as e:
            self.logger.error(f"Error optimizing processing: {e}")
            return {
                "error": str(e),
                "total_optimization_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from optimization history"""
        try:
            # Get optimization history from Redis
            optimization_keys = self.redis.keys("optimization:*")

            if not optimization_keys:
                return {"message": "No optimization history found"}

            # Analyze optimization patterns
            chunking_strategies = {}
            navigation_strategies = {}
            avg_performance = {}

            for key in optimization_keys[:100]:  # Limit to last 100 optimizations
                try:
                    result = json.loads(self.redis.get(key))

                    # Analyze chunking strategies
                    chunking_strategy = result.get("chunking_optimization", {}).get(
                        "strategy_name", "unknown"
                    )
                    chunking_strategies[chunking_strategy] = (
                        chunking_strategies.get(chunking_strategy, 0) + 1
                    )

                    # Analyze navigation strategies
                    navigation_strategy = result.get("navigation_optimization", {}).get(
                        "strategy_name", "unknown"
                    )
                    navigation_strategies[navigation_strategy] = (
                        navigation_strategies.get(navigation_strategy, 0) + 1
                    )

                    # Analyze performance
                    chunking_score = result.get("chunking_optimization", {}).get(
                        "overall_score", 0
                    )
                    navigation_score = result.get("navigation_optimization", {}).get(
                        "overall_score", 0
                    )

                    avg_performance["chunking"] = (
                        avg_performance.get("chunking", 0) + chunking_score
                    )
                    avg_performance["navigation"] = (
                        avg_performance.get("navigation", 0) + navigation_score
                    )

                except Exception as e:
                    self.logger.error(f"Error analyzing optimization key {key}: {e}")
                    continue

            # Calculate averages
            total_optimizations = len(optimization_keys[:100])
            if total_optimizations > 0:
                avg_performance["chunking"] /= total_optimizations
                avg_performance["navigation"] /= total_optimizations

            insights = {
                "total_optimizations": len(optimization_keys),
                "recent_optimizations": total_optimizations,
                "most_used_chunking_strategy": max(
                    chunking_strategies, key=chunking_strategies.get
                )
                if chunking_strategies
                else "unknown",
                "most_used_navigation_strategy": max(
                    navigation_strategies, key=navigation_strategies.get
                )
                if navigation_strategies
                else "unknown",
                "average_performance": avg_performance,
                "strategy_distribution": {
                    "chunking": chunking_strategies,
                    "navigation": navigation_strategies,
                },
            }

            return insights

        except Exception as e:
            self.logger.error(f"Error getting optimization insights: {e}")
            return {"error": str(e)}

    def clear_optimization_cache(self) -> bool:
        """Clear optimization cache"""
        try:
            optimization_keys = self.redis.keys("optimization:*")
            if optimization_keys:
                self.redis.delete(*optimization_keys)
                self.logger.info(
                    f"Cleared {len(optimization_keys)} optimization cache entries"
                )
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error clearing optimization cache: {e}")
            return False
