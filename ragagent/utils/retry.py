"""
Enhanced retry mechanisms and error handling for the Agentic RAG system
"""

import logging
import time
import random
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import asyncio
import traceback
from contextlib import asynccontextmanager

from ..models import ProcessingRequest, ProcessingResponse, APIResponse, ErrorDetail

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur"""

    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_errors: List[Type[Exception]] = None

    def __post_init__(self):
        if self.retry_on_errors is None:
            self.retry_on_errors = [
                ConnectionError,
                TimeoutError,
                asyncio.TimeoutError,
                # Add more specific error types as needed
            ]


@dataclass
class ErrorContext:
    """Context information about an error"""

    error_type: ErrorType
    exception: Exception
    attempt: int
    total_attempts: int
    request: Optional[ProcessingRequest] = None
    timestamp: float = None
    stack_trace: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.stack_trace is None:
            self.stack_trace = traceback.format_exc()


class ErrorHandler:
    """Centralized error handling and recovery"""

    def __init__(self):
        self.error_stats: Dict[ErrorType, int] = {}
        self.recovery_strategies: Dict[ErrorType, Callable] = {}
        self.setup_default_strategies()

    def setup_default_strategies(self):
        """Setup default recovery strategies"""
        self.recovery_strategies[ErrorType.NETWORK_ERROR] = self._handle_network_error
        self.recovery_strategies[ErrorType.TIMEOUT_ERROR] = self._handle_timeout_error
        self.recovery_strategies[ErrorType.RATE_LIMIT_ERROR] = (
            self._handle_rate_limit_error
        )
        self.recovery_strategies[ErrorType.API_ERROR] = self._handle_api_error
        self.recovery_strategies[ErrorType.VALIDATION_ERROR] = (
            self._handle_validation_error
        )
        self.recovery_strategies[ErrorType.PROCESSING_ERROR] = (
            self._handle_processing_error
        )
        self.recovery_strategies[ErrorType.UNKNOWN_ERROR] = self._handle_unknown_error

    def classify_error(self, exception: Exception) -> ErrorType:
        """Classify an exception into an error type"""
        error_type = str(type(exception)).lower()
        message = str(exception).lower()

        # Network-related errors
        if any(keyword in error_type for keyword in ["connection", "network", "http"]):
            return ErrorType.NETWORK_ERROR

        # Timeout errors
        if any(keyword in error_type for keyword in ["timeout", "read timeout"]):
            return ErrorType.TIMEOUT_ERROR

        # Rate limiting errors
        if any(
            keyword in message for keyword in ["rate limit", "too many requests", "429"]
        ):
            return ErrorType.RATE_LIMIT_ERROR

        # API errors
        if any(keyword in error_type for keyword in ["api", "openai", "ollama"]):
            return ErrorType.API_ERROR

        # Validation errors
        if any(
            keyword in error_type for keyword in ["validation", "pydantic", "schema"]
        ):
            return ErrorType.VALIDATION_ERROR

        # Processing errors
        if any(keyword in error_type for keyword in ["processing", "parsing", "chunk"]):
            return ErrorType.PROCESSING_ERROR

        # Unknown error
        return ErrorType.UNKNOWN_ERROR

    def handle_error(self, context: ErrorContext) -> Optional[APIResponse]:
        """Handle an error with appropriate recovery strategy"""
        error_type = context.error_type

        # Update error statistics
        self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1

        logger.error(f"Error occurred: {error_type.value} - {context.exception}")
        logger.debug(f"Stack trace: {context.stack_trace}")

        # Get recovery strategy
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](context)
        else:
            return self._handle_unknown_error(context)

    def _handle_network_error(self, context: ErrorContext) -> Optional[APIResponse]:
        """Handle network errors"""
        logger.info("Network error detected, implementing recovery...")

        # Could implement:
        # - Retry with different endpoint
        # - Fallback to cached response
        # - Use lighter model

        return APIResponse(
            success=False,
            error=ErrorDetail(
                error="NetworkError",
                message=f"Network error occurred: {context.exception}",
                details={
                    "error_type": context.error_type.value,
                    "attempt": context.attempt,
                },
            ),
        )

    def _handle_timeout_error(self, context: ErrorContext) -> Optional[APIResponse]:
        """Handle timeout errors"""
        logger.info("Timeout error detected, implementing recovery...")

        # Could implement:
        # - Retry with longer timeout
        # - Use faster model
        # - Simplified processing

        return APIResponse(
            success=False,
            error=ErrorDetail(
                error="TimeoutError",
                message=f"Request timed out: {context.exception}",
                details={
                    "error_type": context.error_type.value,
                    "attempt": context.attempt,
                },
            ),
        )

    def _handle_rate_limit_error(self, context: ErrorContext) -> Optional[APIResponse]:
        """Handle rate limit errors"""
        logger.info("Rate limit error detected, implementing recovery...")

        # Could implement:
        # - Exponential backoff
        # - Switch to different model
        # - Use cached response

        return APIResponse(
            success=False,
            error=ErrorDetail(
                error="RateLimitError",
                message=f"Rate limit exceeded: {context.exception}",
                details={
                    "error_type": context.error_type.value,
                    "attempt": context.attempt,
                },
            ),
        )

    def _handle_api_error(self, context: ErrorContext) -> Optional[APIResponse]:
        """Handle API errors"""
        logger.info("API error detected, implementing recovery...")

        # Could implement:
        # - Retry with different parameters
        # - Use fallback model
        # - Simplified request

        return APIResponse(
            success=False,
            error=ErrorDetail(
                error="APIError",
                message=f"API error occurred: {context.exception}",
                details={
                    "error_type": context.error_type.value,
                    "attempt": context.attempt,
                },
            ),
        )

    def _handle_validation_error(self, context: ErrorContext) -> Optional[APIResponse]:
        """Handle validation errors"""
        logger.info("Validation error detected, implementing recovery...")

        # Could implement:
        # - Fix request parameters
        # - Use simplified validation
        # - Return partial result

        return APIResponse(
            success=False,
            error=ErrorDetail(
                error="ValidationError",
                message=f"Validation error occurred: {context.exception}",
                details={
                    "error_type": context.error_type.value,
                    "attempt": context.attempt,
                },
            ),
        )

    def _handle_processing_error(self, context: ErrorContext) -> Optional[APIResponse]:
        """Handle processing errors"""
        logger.info("Processing error detected, implementing recovery...")

        # Could implement:
        # - Retry with different chunking
        # - Use simpler processing
        # - Return partial result

        return APIResponse(
            success=False,
            error=ErrorDetail(
                error="ProcessingError",
                message=f"Processing error occurred: {context.exception}",
                details={
                    "error_type": context.error_type.value,
                    "attempt": context.attempt,
                },
            ),
        )

    def _handle_unknown_error(self, context: ErrorContext) -> Optional[APIResponse]:
        """Handle unknown errors"""
        logger.warning("Unknown error detected, no specific recovery available...")

        return APIResponse(
            success=False,
            error=ErrorDetail(
                error="UnknownError",
                message=f"Unknown error occurred: {context.exception}",
                details={
                    "error_type": context.error_type.value,
                    "attempt": context.attempt,
                },
            ),
        )

    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        return {
            error_type.value: count for error_type, count in self.error_stats.items()
        }


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter"""

    def __init__(self, config: RetryConfig):
        self.config = config

    def should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should be retried"""
        return any(
            isinstance(exception, error_type)
            for error_type in self.config.retry_on_errors
        )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter to avoid thundering herd
        if self.config.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        return delay

    async def retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Retry an async function with exponential backoff"""
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if not self.should_retry(e) or attempt == self.config.max_attempts:
                    # Don't retry this exception or reached max attempts
                    raise

                # Calculate delay
                delay = self.get_delay(attempt)

                logger.warning(
                    f"Attempt {attempt} failed, retrying in {delay:.2f}s: {e}"
                )

                # Wait before retry
                await asyncio.sleep(delay)

        # If we get here, all retries failed
        raise last_exception

    def retry_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Retry a sync function with exponential backoff"""
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if not self.should_retry(e) or attempt == self.config.max_attempts:
                    # Don't retry this exception or reached max attempts
                    raise

                # Calculate delay
                delay = self.get_delay(attempt)

                logger.warning(
                    f"Attempt {attempt} failed, retrying in {delay:.2f}s: {e}"
                )

                # Wait before retry
                time.sleep(delay)

        # If we get here, all retries failed
        raise last_exception


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for adding retry logic to functions"""
    if config is None:
        config = RetryConfig()

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                retry_handler = RetryHandler(config)
                return await retry_handler.retry_async(func, *args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                retry_handler = RetryHandler(config)
                return retry_handler.retry_sync(func, *args, **kwargs)

            return sync_wrapper

    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs):
        """Call the wrapped function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return time.time() - self.last_failure_time > self.recovery_timeout

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


def with_circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """Decorator for adding circuit breaker pattern"""

    def decorator(func):
        circuit_breaker = CircuitBreaker(failure_threshold, recovery_timeout)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler"""
    return global_error_handler


def reset_error_stats():
    """Reset error statistics"""
    global_error_handler.error_stats.clear()


def get_error_stats() -> Dict[str, int]:
    """Get global error statistics"""
    return global_error_handler.get_error_stats()
