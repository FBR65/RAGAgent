"""
Utility modules for the Agentic RAG system
"""

from .cache import CacheManager, MemoryCache, DiskCache, CacheEntry, cache_results
from .hash_utils import get_file_hash, get_text_hash, get_directory_hash
from .pool import (
    ConnectionPool,
    RequestPool,
    RequestPoolManager,
    PooledRequest,
    RequestPriority,
    pooled_request_processing,
)
from .retry import (
    ErrorHandler,
    RetryHandler,
    RetryConfig,
    ErrorContext,
    ErrorType,
    CircuitBreaker,
    with_retry,
    with_circuit_breaker,
    global_error_handler,
    get_error_handler,
    reset_error_stats,
    get_error_stats,
)
from .validation import (
    ConfigValidator,
    ConfigManager,
    ValidationResult,
    ValidationRule,
    RequiredRule,
    TypeRule,
    RangeRule,
    RegexRule,
    EnumRule,
    CustomRule,
    SchemaValidator,
    ValidationSeverity,
    get_config_manager,
    validate_config,
)

__all__ = [
    # Cache
    "CacheManager",
    "MemoryCache",
    "DiskCache",
    "CacheEntry",
    "cache_results",
    # Hash Utils
    "get_file_hash",
    "get_text_hash",
    "get_directory_hash",
    # Pool
    "ConnectionPool",
    "RequestPool",
    "RequestPoolManager",
    "PooledRequest",
    "RequestPriority",
    "pooled_request_processing",
    # Retry
    "ErrorHandler",
    "RetryHandler",
    "RetryConfig",
    "ErrorContext",
    "ErrorType",
    "CircuitBreaker",
    "with_retry",
    "with_circuit_breaker",
    "global_error_handler",
    "get_error_handler",
    "reset_error_stats",
    "get_error_stats",
    # Validation
    "ConfigValidator",
    "ConfigManager",
    "ValidationResult",
    "ValidationRule",
    "RequiredRule",
    "TypeRule",
    "RangeRule",
    "RegexRule",
    "EnumRule",
    "CustomRule",
    "SchemaValidator",
    "ValidationSeverity",
    "get_config_manager",
    "validate_config",
]
