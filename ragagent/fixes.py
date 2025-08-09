"""
Comprehensive fixes for identified weaknesses in the RAG system
"""

import logging
import time
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from enum import Enum
import redis
import pickle
from functools import wraps
import traceback

from .models import (
    ProcessingRequest,
    ProcessingResponse,
    PipelineConfig,
    AgentConfig,
    DocumentProcessingConfig,
    Chunk,
    AnswerWithCitations,
    VerificationResult,
    ErrorDetail,
    APIResponse,
    DocumentType,
)
from .config import get_config
from .utils import CacheManager

logger = logging.getLogger(__name__)


class FixType(Enum):
    """Types of fixes to apply"""

    ERROR_HANDLING = "error_handling"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    SECURITY = "security"
    USABILITY = "usability"
    SCALABILITY = "scalability"


@dataclass
class FixResult:
    """Result of applying a fix"""

    fix_name: str
    fix_type: FixType
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    improvements: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class ErrorHandler:
    """Enhanced error handling with recovery mechanisms"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Error handling configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        self.circuit_breaker_threshold = self.config.get("circuit_breaker_threshold", 5)
        self.circuit_breaker_timeout = self.config.get("circuit_breaker_timeout", 60)

        # Circuit breaker state
        self.circuit_breaker_state = {}
        self.error_counts = {}

        # Error classification
        self.classifier = ErrorClassifier()

        self.logger.info("Enhanced error handler initialized")

    def handle_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> FixResult:
        """Handle error with appropriate recovery strategy"""
        start_time = time.time()

        try:
            # Classify error
            error_type = self.classifier.classify_error(error)

            # Log error with context
            self.logger.error(f"Error occurred: {error_type} - {str(error)}")
            if context:
                self.logger.error(f"Error context: {context}")

            # Update error counts
            error_key = f"{error_type}_{context.get('operation', 'unknown')}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

            # Check circuit breaker
            if self._should_trigger_circuit_breaker(error_key):
                self._trigger_circuit_breaker(error_key)
                return FixResult(
                    fix_name="circuit_breaker",
                    fix_type=FixType.RELIABILITY,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Circuit breaker triggered",
                )

            # Apply appropriate recovery strategy
            recovery_strategy = self._get_recovery_strategy(error_type, context)

            if recovery_strategy:
                recovery_result = recovery_strategy(error, context)

                return FixResult(
                    fix_name=f"error_recovery_{error_type}",
                    fix_type=FixType.RELIABILITY,
                    success=recovery_result.get("success", False),
                    execution_time=time.time() - start_time,
                    error_message=recovery_result.get("error_message"),
                    improvements=recovery_result.get("improvements", {}),
                )
            else:
                return FixResult(
                    fix_name="error_handling",
                    fix_type=FixType.ERROR_HANDLING,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Unhandled error type: {error_type}",
                )

        except Exception as e:
            self.logger.error(f"Error in error handler: {e}")
            return FixResult(
                fix_name="error_handler_failure",
                fix_type=FixType.ERROR_HANDLING,
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    def _should_trigger_circuit_breaker(self, error_key: str) -> bool:
        """Check if circuit breaker should be triggered"""
        error_count = self.error_counts.get(error_key, 0)
        threshold = self.circuit_breaker_threshold

        # Check if circuit breaker is already open
        if error_key in self.circuit_breaker_state:
            state = self.circuit_breaker_state[error_key]
            if state["state"] == "open" and time.time() < state["timeout"]:
                return True
            elif state["state"] == "open" and time.time() >= state["timeout"]:
                # Reset circuit breaker
                self.circuit_breaker_state[error_key] = {
                    "state": "closed",
                    "timeout": 0,
                }

        return error_count >= threshold

    def _trigger_circuit_breaker(self, error_key: str):
        """Trigger circuit breaker for error key"""
        self.circuit_breaker_state[error_key] = {
            "state": "open",
            "timeout": time.time() + self.circuit_breaker_timeout,
        }
        self.logger.warning(f"Circuit breaker triggered for {error_key}")

    def _get_recovery_strategy(self, error_type: str, context: Dict[str, Any] = None):
        """Get recovery strategy for error type"""
        recovery_strategies = {
            "network_error": self._recover_from_network_error,
            "timeout_error": self._recover_from_timeout_error,
            "rate_limit_error": self._recover_from_rate_limit_error,
            "memory_error": self._recover_from_memory_error,
            "parsing_error": self._recover_from_parsing_error,
            "validation_error": self._recover_from_validation_error,
            "configuration_error": self._recover_from_configuration_error,
            "unknown_error": self._recover_from_unknown_error,
        }

        return recovery_strategies.get(error_type)

    def _recover_from_network_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Recover from network errors"""
        operation = context.get("operation", "unknown")

        if operation == "document_processing":
            return {
                "success": False,
                "error_message": "Network error during document processing",
                "improvements": {
                    "suggestion": "Check network connectivity and try again",
                    "fallback": "Use cached version if available",
                },
            }

        elif operation == "ai_inference":
            return {
                "success": False,
                "error_message": "Network error during AI inference",
                "improvements": {
                    "suggestion": "Retry with exponential backoff",
                    "fallback": "Use local model if available",
                },
            }

        return {"success": False, "error_message": "Network error recovery failed"}

    def _recover_from_timeout_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Recover from timeout errors"""
        operation = context.get("operation", "unknown")

        if operation == "ai_inference":
            return {
                "success": False,
                "error_message": "Timeout during AI inference",
                "improvements": {
                    "suggestion": "Increase timeout limit",
                    "fallback": "Use smaller model or reduce input size",
                },
            }

        elif operation == "document_processing":
            return {
                "success": False,
                "error_message": "Timeout during document processing",
                "improvements": {
                    "suggestion": "Increase timeout or process document in chunks",
                    "fallback": "Use faster parser",
                },
            }

        return {"success": False, "error_message": "Timeout error recovery failed"}

    def _recover_from_rate_limit_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Recover from rate limit errors"""
        operation = context.get("operation", "unknown")

        if operation == "ai_inference":
            return {
                "success": False,
                "error_message": "Rate limit exceeded",
                "improvements": {
                    "suggestion": "Wait and retry with exponential backoff",
                    "fallback": "Use local model or reduce request frequency",
                },
            }

        return {"success": False, "error_message": "Rate limit error recovery failed"}

    def _recover_from_memory_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Recover from memory errors"""
        operation = context.get("operation", "unknown")

        if operation == "document_processing":
            return {
                "success": False,
                "error_message": "Memory error during document processing",
                "improvements": {
                    "suggestion": "Process document in smaller chunks",
                    "fallback": "Use streaming processing",
                },
            }

        elif operation == "ai_inference":
            return {
                "success": False,
                "error_message": "Memory error during AI inference",
                "improvements": {
                    "suggestion": "Reduce input size or use smaller model",
                    "fallback": "Use batch processing",
                },
            }

        return {"success": False, "error_message": "Memory error recovery failed"}

    def _recover_from_parsing_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Recover from parsing errors"""
        operation = context.get("operation", "unknown")

        if operation == "document_parsing":
            return {
                "success": False,
                "error_message": "Parsing error",
                "improvements": {
                    "suggestion": "Try different parser or clean document",
                    "fallback": "Use text-only extraction",
                },
            }

        return {"success": False, "error_message": "Parsing error recovery failed"}

    def _recover_from_validation_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Recover from validation errors"""
        operation = context.get("operation", "unknown")

        if operation == "input_validation":
            return {
                "success": False,
                "error_message": "Validation error",
                "improvements": {
                    "suggestion": "Check input format and requirements",
                    "fallback": "Use default values",
                },
            }

        return {"success": False, "error_message": "Validation error recovery failed"}

    def _recover_from_configuration_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Recover from configuration errors"""
        operation = context.get("operation", "unknown")

        if operation == "initialization":
            return {
                "success": False,
                "error_message": "Configuration error",
                "improvements": {
                    "suggestion": "Check configuration file and environment variables",
                    "fallback": "Use default configuration",
                },
            }

        return {
            "success": False,
            "error_message": "Configuration error recovery failed",
        }

    def _recover_from_unknown_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Recover from unknown errors"""
        return {
            "success": False,
            "error_message": "Unknown error occurred",
            "improvements": {
                "suggestion": "Check logs and system status",
                "fallback": "Restart service or contact support",
            },
        }

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": self.error_counts,
            "circuit_breaker_state": self.circuit_breaker_state,
            "recovery_attempts": len(
                [k for k, v in self.error_counts.items() if v > 0]
            ),
        }


class ErrorClassifier:
    """Classify errors into appropriate categories"""

    def classify_error(self, error: Exception) -> str:
        """Classify error into appropriate category"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Network errors
        if any(
            keyword in error_str
            for keyword in ["network", "connection", "timeout", "unreachable"]
        ):
            return "network_error"

        # Rate limit errors
        if any(
            keyword in error_str
            for keyword in ["rate limit", "too many requests", "quota"]
        ):
            return "rate_limit_error"

        # Memory errors
        if any(
            keyword in error_str
            for keyword in ["memory", "out of memory", "allocation"]
        ):
            return "memory_error"

        # Timeout errors
        if any(keyword in error_str for keyword in ["timeout", "timed out"]):
            return "timeout_error"

        # Parsing errors
        if any(
            keyword in error_str for keyword in ["parse", "syntax", "format", "invalid"]
        ):
            return "parsing_error"

        # Validation errors
        if any(
            keyword in error_str for keyword in ["validation", "invalid", "required"]
        ):
            return "validation_error"

        # Configuration errors
        if any(
            keyword in error_str for keyword in ["config", "setting", "environment"]
        ):
            return "configuration_error"

        # Default to unknown
        return "unknown_error"


class PerformanceOptimizer:
    """Optimize system performance"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Performance optimization configuration
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)
        self.max_cache_size = self.config.get("max_cache_size", 1000)
        self.enable_compression = self.config.get("enable_compression", True)
        self.enable_parallel_processing = self.config.get(
            "enable_parallel_processing", True
        )

        # Performance metrics
        self.performance_metrics = {}

        # Cache manager
        self.cache = CacheManager()

        # Redis client for distributed caching
        self.redis = redis.Redis(
            host=self.config.get("redis_host", "localhost"),
            port=self.config.get("redis_port", 6379),
        )

        self.logger.info("Performance optimizer initialized")

    def optimize_processing(self, request: ProcessingRequest) -> ProcessingResponse:
        """Optimize processing performance"""
        start_time = time.time()

        try:
            # Check cache first
            if self.enable_caching:
                cached_result = self._get_cached_result(request)
                if cached_result:
                    self.logger.info("Using cached result")
                    return cached_result

            # Apply performance optimizations
            optimized_request = self._optimize_request(request)

            # Process with optimizations
            result = self._process_with_optimizations(optimized_request)

            # Cache result
            if self.enable_caching and result.success:
                self._cache_result(request, result)

            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, result.success)

            return result

        except Exception as e:
            self.logger.error(f"Error in performance optimization: {e}")
            return ProcessingResponse(
                question=request.question,
                answer="Error processing request",
                citations=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error=str(e),
            )

    def _get_cached_result(
        self, request: ProcessingRequest
    ) -> Optional[ProcessingResponse]:
        """Get cached result for request"""
        try:
            # Create cache key
            cache_key = self._create_cache_key(request)

            # Try Redis cache first
            cached_data = self.redis.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)

            # Try local cache
            return self.cache.get(cache_key)

        except Exception as e:
            self.logger.error(f"Error getting cached result: {e}")
            return None

    def _cache_result(self, request: ProcessingRequest, result: ProcessingResponse):
        """Cache processing result"""
        try:
            cache_key = self._create_cache_key(request)

            # Compress result if enabled
            if self.enable_compression:
                result_data = pickle.dumps(result)
                result_data = self._compress_data(result_data)
            else:
                result_data = pickle.dumps(result)

            # Store in Redis with TTL
            self.redis.setex(cache_key, self.cache_ttl, result_data)

            # Also store in local cache
            self.cache.set(cache_key, result, ttl=self.cache_ttl)

        except Exception as e:
            self.logger.error(f"Error caching result: {e}")

    def _create_cache_key(self, request: ProcessingRequest) -> str:
        """Create cache key for request"""
        # Use hash of question and document content
        import hashlib

        key_data = (
            f"{request.question}_{request.document_path}_{hash(str(request.config))}"
        )
        return hashlib.md5(key_data.encode()).hexdigest()

    def _optimize_request(self, request: ProcessingRequest) -> ProcessingRequest:
        """Optimize request for better performance"""
        # Create optimized copy
        optimized_request = ProcessingRequest(
            question=request.question,
            document_path=request.document_path,
            document_type=request.document_type,
            config=request.config,
        )

        # Apply request optimizations
        if optimized_request.config:
            # Optimize chunk size based on document length
            if hasattr(optimized_request.config, "document_processing"):
                doc_size = self._estimate_document_size(optimized_request.document_path)
                if doc_size > 1000000:  # Large document
                    optimized_request.config.document_processing.max_chunk_size = 3000
                elif doc_size < 100000:  # Small document
                    optimized_request.config.document_processing.max_chunk_size = 1000

        return optimized_request

    def _estimate_document_size(self, document_path: str) -> int:
        """Estimate document size"""
        try:
            path = Path(document_path)
            if path.exists():
                return path.stat().st_size
            return 100000  # Default size
        except Exception:
            return 100000  # Default size

    def _process_with_optimizations(
        self, request: ProcessingRequest
    ) -> ProcessingResponse:
        """Process request with performance optimizations"""
        # This would integrate with the main pipeline
        # For now, return a placeholder
        return ProcessingResponse(
            question=request.question,
            answer="Processed with optimizations",
            citations=[],
            confidence_score=0.8,
            processing_time=0.1,
        )

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data for storage"""
        import gzip

        return gzip.compress(data)

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data"""
        import gzip

        return gzip.decompress(data)

    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update performance metrics"""
        if success not in self.performance_metrics:
            self.performance_metrics[success] = []

        self.performance_metrics[success].append(processing_time)

        # Keep only recent metrics (last 100)
        if len(self.performance_metrics[success]) > 100:
            self.performance_metrics[success] = self.performance_metrics[success][-100:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {}

        for success, times in self.performance_metrics.items():
            if times:
                metrics[f"avg_time_{success}"] = sum(times) / len(times)
                metrics[f"max_time_{success}"] = max(times)
                metrics[f"min_time_{success}"] = min(times)
                metrics[f"count_{success}"] = len(times)

        return metrics

    def clear_cache(self):
        """Clear all caches"""
        try:
            # Clear Redis cache
            self.redis.flushdb()

            # Clear local cache
            self.cache.clear()

            self.logger.info("Cache cleared successfully")

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")


class ReliabilityEnhancer:
    """Enhance system reliability"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Reliability configuration
        self.health_check_interval = self.config.get("health_check_interval", 30)
        self.backup_enabled = self.config.get("backup_enabled", True)
        self.backup_interval = self.config.get("backup_interval", 3600)
        self.monitoring_enabled = self.config.get("monitoring_enabled", True)

        # Health monitoring
        self.health_status = {}
        self.last_health_check = {}

        # Backup system
        self.backup_manager = BackupManager()

        # Monitoring system
        self.monitoring = MonitoringSystem()

        self.logger.info("Reliability enhancer initialized")

    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        start_time = time.time()

        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {},
            "issues": [],
        }

        try:
            # Check system components
            checks = {
                "redis_connection": self._check_redis_connection,
                "document_parsers": self._check_document_parsers,
                "ai_models": self._check_ai_models,
                "storage_system": self._check_storage_system,
                "network_connectivity": self._check_network_connectivity,
            }

            for check_name, check_func in checks.items():
                try:
                    result = check_func()
                    health_status["checks"][check_name] = result

                    if not result["healthy"]:
                        health_status["issues"].append(result["issue"])
                        health_status["overall_status"] = "degraded"

                except Exception as e:
                    health_status["checks"][check_name] = {
                        "healthy": False,
                        "issue": f"Health check failed: {str(e)}",
                    }
                    health_status["issues"].append(f"Health check failed: {str(e)}")
                    health_status["overall_status"] = "unhealthy"

            # Update last health check
            self.last_health_check = health_status

            # Log health status
            self.logger.info(
                f"Health check completed: {health_status['overall_status']}"
            )

            return health_status

        except Exception as e:
            self.logger.error(f"Error during health check: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "unhealthy",
                "checks": {},
                "issues": [f"Health check failed: {str(e)}"],
            }

    def _check_redis_connection(self) -> Dict[str, Any]:
        """Check Redis connection"""
        try:
            self.redis.ping()
            return {"healthy": True, "message": "Redis connection OK"}
        except Exception as e:
            return {"healthy": False, "issue": f"Redis connection failed: {str(e)}"}

    def _check_document_parsers(self) -> Dict[str, Any]:
        """Check document parsers"""
        try:
            from .parsers import ParserRegistry

            registry = ParserRegistry()
            parsers = registry.get_available_parsers()

            if parsers:
                return {"healthy": True, "message": f"Available parsers: {parsers}"}
            else:
                return {"healthy": False, "issue": "No parsers available"}

        except Exception as e:
            return {"healthy": False, "issue": f"Parsers check failed: {str(e)}"}

    def _check_ai_models(self) -> Dict[str, Any]:
        """Check AI models"""
        try:
            from .openai_client import OpenAIClientFactory

            client = OpenAIClientFactory.create_client()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )

            return {"healthy": True, "message": "AI model connection OK"}

        except Exception as e:
            return {"healthy": False, "issue": f"AI model check failed: {str(e)}"}

    def _check_storage_system(self) -> Dict[str, Any]:
        """Check storage system"""
        try:
            # Check if we can write and read files
            test_file = Path("test_health_check.tmp")

            # Write test file
            with open(test_file, "w") as f:
                f.write("test")

            # Read test file
            with open(test_file, "r") as f:
                content = f.read()

            # Clean up
            test_file.unlink()

            if content == "test":
                return {"healthy": True, "message": "Storage system OK"}
            else:
                return {"healthy": False, "issue": "Storage system read/write failed"}

        except Exception as e:
            return {"healthy": False, "issue": f"Storage system check failed: {str(e)}"}

    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity"""
        try:
            import requests

            # Test basic connectivity
            response = requests.get("https://www.google.com", timeout=5)

            if response.status_code == 200:
                return {"healthy": True, "message": "Network connectivity OK"}
            else:
                return {
                    "healthy": False,
                    "issue": f"Network connectivity issue: {response.status_code}",
                }

        except Exception as e:
            return {
                "healthy": False,
                "issue": f"Network connectivity check failed: {str(e)}",
            }

    def create_backup(self) -> Dict[str, Any]:
        """Create system backup"""
        start_time = time.time()

        try:
            backup_result = self.backup_manager.create_backup()

            return {
                "success": backup_result["success"],
                "backup_id": backup_result.get("backup_id"),
                "backup_size": backup_result.get("backup_size"),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

    def restore_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restore system from backup"""
        start_time = time.time()

        try:
            restore_result = self.backup_manager.restore_backup(backup_id)

            return {
                "success": restore_result["success"],
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error restoring backup: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

    def get_reliability_metrics(self) -> Dict[str, Any]:
        """Get reliability metrics"""
        metrics = {
            "health_status": self.health_status,
            "last_health_check": self.last_health_check,
            "backup_status": self.backup_manager.get_backup_status(),
            "monitoring_metrics": self.monitoring.get_metrics(),
        }

        return metrics


class BackupManager:
    """Manage system backups"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Backup configuration
        self.backup_directory = Path(self.config.get("backup_directory", "backups"))
        self.max_backups = self.config.get("max_backups", 10)
        self.backup_compression = self.config.get("backup_compression", True)

        # Create backup directory if it doesn't exist
        self.backup_directory.mkdir(parents=True, exist_ok=True)

        self.logger.info("Backup manager initialized")

    def create_backup(self) -> Dict[str, Any]:
        """Create system backup"""
        start_time = time.time()

        try:
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backup_directory / backup_id

            # Create backup directory
            backup_path.mkdir(exist_ok=True)

            # Backup configuration
            self._backup_configuration(backup_path)

            # Backup data
            self._backup_data(backup_path)

            # Backup logs
            self._backup_logs(backup_path)

            # Compress backup if enabled
            if self.backup_compression:
                backup_path = self._compress_backup(backup_path)

            # Clean up old backups
            self._cleanup_old_backups()

            backup_size = self._get_directory_size(backup_path)

            self.logger.info(f"Backup created: {backup_id} ({backup_size} bytes)")

            return {
                "success": True,
                "backup_id": backup_id,
                "backup_path": str(backup_path),
                "backup_size": backup_size,
                "execution_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "execution_time": time.time() - start_time,
            }

    def _backup_configuration(self, backup_path: Path):
        """Backup system configuration"""
        try:
            from .config import get_config

            config = get_config()
            config_file = backup_path / "config.json"

            with open(config_file, "w") as f:
                json.dump(config.dict(), f, indent=2)

        except Exception as e:
            self.logger.error(f"Error backing up configuration: {e}")

    def _backup_data(self, backup_path: Path):
        """Backup system data"""
        try:
            # Backup cache data
            cache_path = backup_path / "cache"
            cache_path.mkdir(exist_ok=True)

            # Backup Redis data
            try:
                redis_data = self._get_redis_data()
                if redis_data:
                    with open(cache_path / "redis_data.json", "w") as f:
                        json.dump(redis_data, f, indent=2)
            except Exception as e:
                self.logger.error(f"Error backing up Redis data: {e}")

            # Backup local cache
            try:
                from .utils import CacheManager

                cache = CacheManager()
                local_cache_data = cache.get_all()
                if local_cache_data:
                    with open(cache_path / "local_cache.json", "w") as f:
                        json.dump(local_cache_data, f, indent=2)
            except Exception as e:
                self.logger.error(f"Error backing up local cache: {e}")

        except Exception as e:
            self.logger.error(f"Error backing up data: {e}")

    def _backup_logs(self, backup_path: Path):
        """Backup system logs"""
        try:
            log_files = ["ragagent.log", "error.log"]

            logs_path = backup_path / "logs"
            logs_path.mkdir(exist_ok=True)

            for log_file in log_files:
                log_path = Path(log_file)
                if log_path.exists():
                    target_path = logs_path / log_file
                    import shutil

                    shutil.copy2(log_path, target_path)

        except Exception as e:
            self.logger.error(f"Error backing up logs: {e}")

    def _compress_backup(self, backup_path: Path) -> Path:
        """Compress backup directory"""
        import tarfile

        compressed_path = backup_path.with_suffix(".tar.gz")

        with tarfile.open(compressed_path, "w:gz") as tar:
            tar.add(backup_path, arcname=backup_path.name)

        # Remove original directory
        import shutil

        shutil.rmtree(backup_path)

        return compressed_path

    def _cleanup_old_backups(self):
        """Clean up old backups"""
        try:
            # Get all backup directories
            backup_dirs = [d for d in self.backup_directory.iterdir() if d.is_dir()]

            # Sort by modification time
            backup_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove old backups
            for backup_dir in backup_dirs[self.max_backups :]:
                import shutil

                shutil.rmtree(backup_dir)

        except Exception as e:
            self.logger.error(f"Error cleaning up old backups: {e}")

    def _get_directory_size(self, directory: Path) -> int:
        """Get directory size in bytes"""
        total_size = 0

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)

        return total_size

    def _get_redis_data(self) -> Dict[str, Any]:
        """Get Redis data for backup"""
        try:
            redis_data = {}

            # Get all keys
            keys = self.redis.keys("*")

            for key in keys:
                value = self.redis.get(key)
                if value:
                    redis_data[key.decode()] = pickle.loads(value)

            return redis_data

        except Exception as e:
            self.logger.error(f"Error getting Redis data: {e}")
            return {}

    def restore_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restore system from backup"""
        start_time = time.time()

        try:
            backup_path = self.backup_directory / backup_id

            # Check if backup exists
            if not backup_path.exists():
                # Try compressed version
                backup_path = self.backup_directory / f"{backup_id}.tar.gz"
                if backup_path.exists():
                    backup_path = self._decompress_backup(backup_path)
                else:
                    return {
                        "success": False,
                        "error_message": f"Backup {backup_id} not found",
                    }

            # Restore configuration
            self._restore_configuration(backup_path)

            # Restore data
            self._restore_data(backup_path)

            # Restore logs
            self._restore_logs(backup_path)

            self.logger.info(f"Backup restored: {backup_id}")

            return {"success": True, "execution_time": time.time() - start_time}

        except Exception as e:
            self.logger.error(f"Error restoring backup: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "execution_time": time.time() - start_time,
            }

    def _decompress_backup(self, backup_path: Path) -> Path:
        """Decompress backup"""
        import tarfile

        decompressed_path = backup_path.with_suffix("")
        decompressed_path.mkdir(exist_ok=True)

        with tarfile.open(backup_path, "r:gz") as tar:
            tar.extractall(decompressed_path)

        return decompressed_path / backup_path.stem

    def _restore_configuration(self, backup_path: Path):
        """Restore system configuration"""
        try:
            config_file = backup_path / "config.json"
            if config_file.exists():
                with open(config_file, "r") as f:
                    config_data = json.load(f)

                from .config import PipelineConfig

                config = PipelineConfig(**config_data)

                # Save configuration
                config_manager = ConfigManager()
                config_manager.save_config(config)

        except Exception as e:
            self.logger.error(f"Error restoring configuration: {e}")

    def _restore_data(self, backup_path: Path):
        """Restore system data"""
        try:
            cache_path = backup_path / "cache"
            if cache_path.exists():
                # Restore Redis data
                redis_file = cache_path / "redis_data.json"
                if redis_file.exists():
                    with open(redis_file, "r") as f:
                        redis_data = json.load(f)

                    for key, value in redis_data.items():
                        self.redis.set(key, pickle.dumps(value))

                # Restore local cache
                local_cache_file = cache_path / "local_cache.json"
                if local_cache_file.exists():
                    with open(local_cache_file, "r") as f:
                        local_cache_data = json.load(f)

                    from .utils import CacheManager

                    cache = CacheManager()
                    for key, value in local_cache_data.items():
                        cache.set(key, value)

        except Exception as e:
            self.logger.error(f"Error restoring data: {e}")

    def _restore_logs(self, backup_path: Path):
        """Restore system logs"""
        try:
            logs_path = backup_path / "logs"
            if logs_path.exists():
                for log_file in logs_path.iterdir():
                    if log_file.is_file():
                        target_path = Path(log_file.name)
                        import shutil

                        shutil.copy2(log_file, target_path)

        except Exception as e:
            self.logger.error(f"Error restoring logs: {e}")

    def get_backup_status(self) -> Dict[str, Any]:
        """Get backup status"""
        try:
            backup_dirs = [d for d in self.backup_directory.iterdir() if d.is_dir()]

            backups = []
            for backup_dir in backup_dirs:
                backup_info = {
                    "backup_id": backup_dir.name,
                    "created_time": datetime.fromtimestamp(
                        backup_dir.stat().st_mtime
                    ).isoformat(),
                    "size": self._get_directory_size(backup_dir),
                }
                backups.append(backup_info)

            return {
                "total_backups": len(backups),
                "backups": sorted(
                    backups, key=lambda x: x["created_time"], reverse=True
                ),
                "backup_directory": str(self.backup_directory),
            }

        except Exception as e:
            self.logger.error(f"Error getting backup status: {e}")
            return {"error": str(e)}


class MonitoringSystem:
    """System monitoring and alerting"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Monitoring configuration
        self.metrics_retention = self.config.get("metrics_retention", 86400)  # 24 hours
        self.alert_thresholds = self.config.get(
            "alert_thresholds",
            {
                "cpu_usage": 80,
                "memory_usage": 85,
                "disk_usage": 90,
                "error_rate": 5,
                "response_time": 1000,
            },
        )

        # Metrics storage
        self.metrics = {}
        self.alerts = []

        # System monitoring
        self.system_monitor = SystemMonitor()

        self.logger.info("Monitoring system initialized")

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": self.system_monitor.get_system_metrics(),
                "application_metrics": self.get_application_metrics(),
                "performance_metrics": self.get_performance_metrics(),
            }

            # Store metrics
            self._store_metrics(metrics)

            # Check for alerts
            self._check_alerts(metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return {"error": str(e)}

    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics"""
        metrics = {
            "total_requests": len(self.metrics),
            "active_connections": self._get_active_connections(),
            "cache_hit_rate": self._get_cache_hit_rate(),
            "error_rate": self._get_error_rate(),
        }

        return metrics

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.metrics:
            return {}

        recent_metrics = self._get_recent_metrics(3600)  # Last hour

        if not recent_metrics:
            return {}

        # Calculate performance statistics
        response_times = [m.get("response_time", 0) for m in recent_metrics]
        error_rates = [m.get("error_rate", 0) for m in recent_metrics]

        return {
            "avg_response_time": sum(response_times) / len(response_times)
            if response_times
            else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "avg_error_rate": sum(error_rates) / len(error_rates) if error_rates else 0,
            "total_requests": len(recent_metrics),
        }

    def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics for analysis"""
        timestamp = metrics.get("timestamp", datetime.now().isoformat())

        # Clean up old metrics
        self._cleanup_old_metrics()

        # Store metrics
        self.metrics[timestamp] = metrics

    def _cleanup_old_metrics(self):
        """Clean up old metrics"""
        cutoff_time = datetime.now().timestamp() - self.metrics_retention

        old_metrics = []
        for timestamp, metrics in self.metrics.items():
            try:
                metric_time = datetime.fromisoformat(timestamp).timestamp()
                if metric_time < cutoff_time:
                    old_metrics.append(timestamp)
            except Exception:
                old_metrics.append(timestamp)

        for timestamp in old_metrics:
            del self.metrics[timestamp]

    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions"""
        try:
            system_metrics = metrics.get("system_metrics", {})
            performance_metrics = metrics.get("performance_metrics", {})

            # Check CPU usage
            cpu_usage = system_metrics.get("cpu_usage", 0)
            if cpu_usage > self.alert_thresholds["cpu_usage"]:
                self._create_alert("high_cpu_usage", f"CPU usage is {cpu_usage}%")

            # Check memory usage
            memory_usage = system_metrics.get("memory_usage", 0)
            if memory_usage > self.alert_thresholds["memory_usage"]:
                self._create_alert(
                    "high_memory_usage", f"Memory usage is {memory_usage}%"
                )

            # Check disk usage
            disk_usage = system_metrics.get("disk_usage", 0)
            if disk_usage > self.alert_thresholds["disk_usage"]:
                self._create_alert("high_disk_usage", f"Disk usage is {disk_usage}%")

            # Check error rate
            error_rate = performance_metrics.get("avg_error_rate", 0)
            if error_rate > self.alert_thresholds["error_rate"]:
                self._create_alert("high_error_rate", f"Error rate is {error_rate}%")

            # Check response time
            response_time = performance_metrics.get("avg_response_time", 0)
            if response_time > self.alert_thresholds["response_time"]:
                self._create_alert(
                    "high_response_time", f"Response time is {response_time}ms"
                )

        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")

    def _create_alert(self, alert_type: str, message: str):
        """Create alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,
            "message": message,
            "severity": "warning",
        }

        self.alerts.append(alert)
        self.logger.warning(f"Alert created: {alert_type} - {message}")

    def _get_active_connections(self) -> int:
        """Get number of active connections"""
        # This would implement actual connection counting
        return 0

    def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        # This would implement actual cache hit rate calculation
        return 0.8

    def _get_error_rate(self) -> float:
        """Get error rate"""
        # This would implement actual error rate calculation
        return 0.01

    def _get_recent_metrics(self, seconds: int) -> List[Dict[str, Any]]:
        """Get metrics from the last N seconds"""
        cutoff_time = datetime.now().timestamp() - seconds

        recent_metrics = []
        for timestamp, metrics in self.metrics.items():
            try:
                metric_time = datetime.fromisoformat(timestamp).timestamp()
                if metric_time >= cutoff_time:
                    recent_metrics.append(metrics)
            except Exception:
                continue

        return recent_metrics

    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics"""
        return {
            "metrics": self.metrics,
            "alerts": self.alerts,
            "alert_thresholds": self.alert_thresholds,
            "total_metrics": len(self.metrics),
            "total_alerts": len(self.alerts),
        }


class SystemMonitor:
    """Monitor system metrics"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            import psutil

            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_total = memory.total
            memory_available = memory.available

            # Disk metrics
            disk = psutil.disk_usage("/")
            disk_usage = disk.percent
            disk_total = disk.total
            disk_free = disk.free

            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv

            return {
                "cpu_usage": cpu_usage,
                "cpu_count": cpu_count,
                "memory_usage": memory_usage,
                "memory_total": memory_total,
                "memory_available": memory_available,
                "disk_usage": disk_usage,
                "disk_total": disk_total,
                "disk_free": disk_free,
                "network_bytes_sent": network_bytes_sent,
                "network_bytes_recv": network_bytes_recv,
            }

        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {}


class ConfigManager:
    """Enhanced configuration management"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuration management
        self.config_file = Path(self.config.get("config_file", "config.json"))
        self.config_version = self.config.get("config_version", "1.0.0")

        self.logger.info("Enhanced configuration manager initialized")

    def load_config(self) -> Dict[str, Any]:
        """Load configuration with validation"""
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    config_data = json.load(f)

                # Validate configuration
                if self._validate_config(config_data):
                    self.logger.info("Configuration loaded successfully")
                    return config_data
                else:
                    self.logger.error("Configuration validation failed")
                    return self._get_default_config()
            else:
                self.logger.info("No configuration file found, using defaults")
                return self._get_default_config()

        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()

    def save_config(self, config: Dict[str, Any]):
        """Save configuration with validation"""
        try:
            # Validate configuration
            if not self._validate_config(config):
                raise ValueError("Invalid configuration")

            # Add metadata
            config["metadata"] = {
                "version": self.config_version,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            # Save configuration
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)

            self.logger.info("Configuration saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        try:
            # Basic validation
            required_fields = ["agent", "document_processing", "enable_verification"]

            for field in required_fields:
                if field not in config:
                    self.logger.error(f"Missing required field: {field}")
                    return False

            # Validate agent configuration
            agent_config = config.get("agent", {})
            if not isinstance(agent_config, dict):
                self.logger.error("Agent configuration must be a dictionary")
                return False

            # Validate document processing configuration
            doc_config = config.get("document_processing", {})
            if not isinstance(doc_config, dict):
                self.logger.error(
                    "Document processing configuration must be a dictionary"
                )
                return False

            # Validate numeric fields
            numeric_fields = {
                "agent": ["temperature", "max_tokens", "timeout"],
                "document_processing": [
                    "max_chunk_size",
                    "min_chunk_size",
                    "overlap_tokens",
                ],
            }

            for section, fields in numeric_fields.items():
                section_config = config.get(section, {})
                for field in fields:
                    if field in section_config:
                        try:
                            value = float(section_config[field])
                            if value < 0:
                                self.logger.error(f"Field {field} must be non-negative")
                                return False
                        except (ValueError, TypeError):
                            self.logger.error(f"Field {field} must be a number")
                            return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "agent": {
                "model_name": "qwen3:latest",
                "base_url": "http://localhost:11434",
                "temperature": 0.1,
                "max_tokens": 4000,
                "timeout": 30,
            },
            "document_processing": {
                "max_chunk_size": 2000,
                "min_chunk_size": 100,
                "overlap_tokens": 100,
                "max_initial_chunks": 20,
                "max_navigation_depth": 3,
            },
            "enable_verification": True,
            "enable_caching": True,
            "cache_ttl": 3600,
            "max_cache_size": 1000,
            "config_version": self.config_version,
            "metadata": {
                "version": self.config_version,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            },
        }


class ComprehensiveFixesManager:
    """Manager for applying comprehensive fixes"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize fix components
        self.error_handler = ErrorHandler(config)
        self.performance_optimizer = PerformanceOptimizer(config)
        self.reliability_enhancer = ReliabilityEnhancer(config)

        # Fix tracking
        self.fix_results = []
        self.fix_history = []

        self.logger.info("Comprehensive fixes manager initialized")

    def apply_all_fixes(self) -> List[FixResult]:
        """Apply all fixes and return results"""
        start_time = time.time()

        self.logger.info("Applying comprehensive fixes...")

        fix_results = []

        # Apply error handling fixes
        error_fix_result = self.apply_error_handling_fixes()
        fix_results.append(error_fix_result)

        # Apply performance fixes
        performance_fix_result = self.apply_performance_fixes()
        fix_results.append(performance_fix_result)

        # Apply reliability fixes
        reliability_fix_result = self.apply_reliability_fixes()
        fix_results.append(reliability_fix_result)

        # Track fix application
        self._track_fix_application(fix_results, time.time() - start_time)

        return fix_results

    def apply_error_handling_fixes(self) -> FixResult:
        """Apply error handling fixes"""
        start_time = time.time()

        try:
            # Test error handling
            test_error = Exception("Test error for error handling")
            context = {"operation": "test", "component": "error_handler"}

            result = self.error_handler.handle_error(test_error, context)

            return FixResult(
                fix_name="error_handling_fixes",
                fix_type=FixType.ERROR_HANDLING,
                success=result.success,
                execution_time=time.time() - start_time,
                error_message=result.error_message,
                improvements=result.improvements,
            )

        except Exception as e:
            return FixResult(
                fix_name="error_handling_fixes",
                fix_type=FixType.ERROR_HANDLING,
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    def apply_performance_fixes(self) -> FixResult:
        """Apply performance fixes"""
        start_time = time.time()

        try:
            # Test performance optimization
            test_request = ProcessingRequest(
                question="Test performance optimization",
                document_path="test_document.txt",
            )

            result = self.performance_optimizer.optimize_processing(test_request)

            return FixResult(
                fix_name="performance_fixes",
                fix_type=FixType.PERFORMANCE,
                success=result.success,
                execution_time=time.time() - start_time,
                error_message=getattr(result, "error", None),
                improvements=self.performance_optimizer.get_performance_metrics(),
            )

        except Exception as e:
            return FixResult(
                fix_name="performance_fixes",
                fix_type=FixType.PERFORMANCE,
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    def apply_reliability_fixes(self) -> FixResult:
        """Apply reliability fixes"""
        start_time = time.time()

        try:
            # Test health check
            health_status = self.reliability_enhancer.perform_health_check()

            # Test backup
            backup_result = self.reliability_enhancer.create_backup()

            return FixResult(
                fix_name="reliability_fixes",
                fix_type=FixType.RELIABILITY,
                success=health_status["overall_status"] == "healthy"
                and backup_result["success"],
                execution_time=time.time() - start_time,
                error_message=backup_result.get("error_message"),
                improvements={
                    "health_status": health_status,
                    "backup_status": backup_result,
                },
            )

        except Exception as e:
            return FixResult(
                fix_name="reliability_fixes",
                fix_type=FixType.RELIABILITY,
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    def _track_fix_application(self, fix_results: List[FixResult], total_time: float):
        """Track fix application"""
        application_record = {
            "timestamp": datetime.now().isoformat(),
            "total_time": total_time,
            "fix_results": [result.to_dict() for result in fix_results],
            "summary": {
                "total_fixes": len(fix_results),
                "successful_fixes": len([r for r in fix_results if r.success]),
                "failed_fixes": len([r for r in fix_results if not r.success]),
            },
        }

        self.fix_history.append(application_record)

        # Keep only recent history (last 100 records)
        if len(self.fix_history) > 100:
            self.fix_history = self.fix_history[-100:]

    def get_fix_summary(self) -> Dict[str, Any]:
        """Get fix application summary"""
        if not self.fix_history:
            return {"message": "No fixes applied yet"}

        latest_application = self.fix_history[-1]

        return {
            "latest_application": latest_application,
            "total_applications": len(self.fix_history),
            "success_rate": self._calculate_success_rate(),
            "common_issues": self._identify_common_issues(),
        }

    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        if not self.fix_history:
            return 0.0

        total_fixes = 0
        successful_fixes = 0

        for application in self.fix_history:
            for fix_result in application["fix_results"]:
                total_fixes += 1
                if fix_result["success"]:
                    successful_fixes += 1

        return successful_fixes / total_fixes if total_fixes > 0 else 0.0

    def _identify_common_issues(self) -> List[Dict[str, Any]]:
        """Identify common issues"""
        issues = {}

        for application in self.fix_history:
            for fix_result in application["fix_results"]:
                if not fix_result["success"]:
                    error_message = fix_result.get("error_message", "Unknown error")
                    if error_message not in issues:
                        issues[error_message] = 0
                    issues[error_message] += 1

        # Return top 5 most common issues
        sorted_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)
        return [{"issue": issue, "count": count} for issue, count in sorted_issues[:5]]

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "error_handler": {
                "statistics": self.error_handler.get_error_statistics(),
                "circuit_breaker_state": self.error_handler.circuit_breaker_state,
            },
            "performance_optimizer": {
                "metrics": self.performance_optimizer.get_performance_metrics(),
                "cache_status": "enabled"
                if self.performance_optimizer.enable_caching
                else "disabled",
            },
            "reliability_enhancer": {
                "health_status": self.reliability_enhancer.get_reliability_metrics(),
                "backup_status": self.reliability_enhancer.backup_manager.get_backup_status(),
            },
            "fix_summary": self.get_fix_summary(),
        }
