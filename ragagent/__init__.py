"""
Agentic RAG System

A comprehensive Retrieval Augmented Generation system with AI agents
for intelligent document processing and answer generation.
"""

from .pipeline import AgenticRAGPipeline, create_pipeline, process_document
from .unified_client import (
    UnifiedClient,
    ClientConfig,
    ProviderType,
    ClientFactory,
    get_default_client,
    set_default_client,
)
from .config_versioning import (
    ConfigVersionManager,
    ConfigVersionInfo,
    ConfigMetadata,
    ConfigMigrator,
    ConfigVersion,
    get_version_manager,
    set_version_manager,
)
from .enhanced_timeout_retry import (
    ResilienceManager,
    TimeoutHandler,
    RetryHandler,
    CircuitBreaker,
    RateLimiter,
    TimeoutConfig,
    RetryConfig,
    CircuitBreakerConfig,
    RateLimitConfig,
    RetryStrategy,
    CircuitState,
    resilient,
    with_circuit_breaker,
    rate_limited,
    get_resilience_manager,
    set_resilience_manager,
)
from .hybrid_rag import HybridRAGSystem, SearchStrategy, BGEEmbedder, QdrantVectorDB
from .specialized_agents import (
    SpecializedAgentFactory,
    LegalAgent,
    MedicalAgent,
    TechnicalAgent,
    DomainType,
    SpecializedAgentConfig,
    DomainKnowledge,
)
from .learning_system import (
    LearningSystemManager,
    AdaptiveProcessor,
    PreferenceLearner,
    LearningProfile,
    UserPreference,
    Interaction,
)
from .distributed_architecture import (
    DistributedRAGSystem,
    TaskScheduler,
    ComputeNode,
    DistributedTask,
    NodeInfo,
    NodeStatus,
    TaskPriority,
)
from .ai_optimization import (
    AIOptimizationManager,
    ChunkingOptimizer,
    NavigationOptimizer,
    DocumentAnalyzer,
    ChunkingStrategy,
    NavigationStrategy,
    OptimizationResult,
    ChunkingConfig,
    NavigationConfig,
)
from .fixes import (
    ComprehensiveFixesManager,
    ErrorHandler,
    PerformanceOptimizer,
    ReliabilityEnhancer,
    BackupManager,
    MonitoringSystem,
    SystemMonitor,
    ConfigManager,
    FixResult,
    FixType,
)
from .ocr_support import (
    OCRSupport,
    OCRParser,
    OCRProcessor,
    OCRPreprocessor,
    OCRPostprocessor,
    LayoutAnalyzer,
    OCRConfig,
    OCRResult,
    OCRMode,
    OCRQuality,
)
from .models import (
    ProcessingRequest,
    ProcessingResponse,
    PipelineConfig,
    AgentConfig,
    DocumentProcessingConfig,
    Chunk,
    AnswerWithCitations,
    VerificationResult,
    VerificationConfidence,
    ErrorDetail,
    APIResponse,
    DocumentType,
)
from .cli import main as cli_main
from .api import app as api_app

__version__ = "0.1.0"
__author__ = "Frank Bernhard Reis"
__email__ = "fbr65@duck.com"

__all__ = [
    # Pipeline
    "AgenticRAGPipeline",
    "create_pipeline",
    "process_document",
    # Unified Client
    "UnifiedClient",
    "ClientConfig",
    "ProviderType",
    "ClientFactory",
    "get_default_client",
    "set_default_client",
    # Hybrid RAG
    "HybridRAGSystem",
    "SearchStrategy",
    "BGEEmbedder",
    "QdrantVectorDB",
    # Specialized Agents
    "SpecializedAgentFactory",
    "LegalAgent",
    "MedicalAgent",
    "TechnicalAgent",
    "DomainType",
    "SpecializedAgentConfig",
    "DomainKnowledge",
    # Learning System
    "LearningSystemManager",
    "AdaptiveProcessor",
    "PreferenceLearner",
    "LearningProfile",
    "UserPreference",
    "Interaction",
    # Distributed Architecture
    "DistributedRAGSystem",
    "TaskScheduler",
    "ComputeNode",
    "DistributedTask",
    "NodeInfo",
    "NodeStatus",
    "TaskPriority",
    # AI Optimization
    "AIOptimizationManager",
    "ChunkingOptimizer",
    "NavigationOptimizer",
    "DocumentAnalyzer",
    "ChunkingStrategy",
    "NavigationStrategy",
    "OptimizationResult",
    "ChunkingConfig",
    "NavigationConfig",
    # Fixes
    "ComprehensiveFixesManager",
    "ErrorHandler",
    "PerformanceOptimizer",
    "ReliabilityEnhancer",
    "BackupManager",
    "MonitoringSystem",
    "SystemMonitor",
    "ConfigManager",
    "FixResult",
    "FixType",
    # OCR Support
    "OCRSupport",
    "OCRParser",
    "OCRProcessor",
    "OCRPreprocessor",
    "OCRPostprocessor",
    "LayoutAnalyzer",
    "OCRConfig",
    "OCRResult",
    "OCRMode",
    "OCRQuality",
    # Enhanced Timeout & Retry
    "ResilienceManager",
    "TimeoutHandler",
    "RetryHandler",
    "CircuitBreaker",
    "RateLimiter",
    "TimeoutConfig",
    "RetryConfig",
    "CircuitBreakerConfig",
    "RateLimitConfig",
    "RetryStrategy",
    "CircuitState",
    "resilient",
    "with_circuit_breaker",
    "rate_limited",
    "get_resilience_manager",
    "set_resilience_manager",
    # Config Versioning
    "ConfigVersionManager",
    "ConfigVersionInfo",
    "ConfigMetadata",
    "ConfigMigrator",
    "ConfigVersion",
    "get_version_manager",
    "set_version_manager",
    # Models
    "ProcessingRequest",
    "ProcessingResponse",
    "PipelineConfig",
    "AgentConfig",
    "DocumentProcessingConfig",
    "Chunk",
    "AnswerWithCitations",
    "VerificationResult",
    "VerificationConfidence",
    "ErrorDetail",
    "APIResponse",
    "DocumentType",
    # CLI
    "cli_main",
    # API
    "api_app",
]
