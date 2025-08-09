"""
Enhanced timeout and retry logic for the RAG system
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Awaitable
from dataclasses import dataclass
from enum import Enum
import random
import functools
from contextlib import asynccontextmanager
import aiohttp
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategies"""

    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    EXPONENTIAL_JITTER = "exponential_jitter"
    ADAPTIVE = "adaptive"


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class TimeoutConfig:
    """Timeout configuration"""

    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    total_timeout: float = 60.0
    pool_timeout: float = 30.0
    pool_connections: int = 100
    pool_maxsize: int = 100
    keepalive_timeout: float = 30.0
    dns_cache: bool = True
    dns_timeout: float = 10.0


@dataclass
class RetryConfig:
    """Retry configuration"""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER
    retry_on_status: List[int] = None
    retry_on_exceptions: List[Exception] = None
    backoff_factor: float = 2.0
    jitter: float = 0.1
    respect_retry_after_header: bool = True

    def __post_init__(self):
        if self.retry_on_status is None:
            self.retry_on_status = [408, 429, 500, 502, 503, 504]
        if self.retry_on_exceptions is None:
            self.retry_on_exceptions = [
                asyncio.TimeoutError,
                aiohttp.ClientError,
                aiohttp.ClientResponseError,
                aiohttp.ServerTimeoutError,
                aiohttp.ClientPayloadError,
                ConnectionError,
                TimeoutError,
            ]


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""

    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: tuple = None
    fallback_function: Optional[Callable] = None

    def __post_init__(self):
        if self.expected_exception is None:
            self.expected_exception = (
                asyncio.TimeoutError,
                aiohttp.ClientError,
                aiohttp.ClientResponseError,
                ConnectionError,
                TimeoutError,
            )


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""

    requests_per_second: int = 10
    requests_per_minute: int = 600
    requests_per_hour: int = 36000
    burst_size: int = 5
    dry_run: bool = False


class TimeoutHandler:
    """Enhanced timeout handler"""

    def __init__(self, config: TimeoutConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Connection pool
        self.connector = aiohttp.TCPConnector(
            limit=config.pool_connections,
            limit_per_host=config.pool_maxsize,
            ttl_dns_cache=config.dns_cache,
            use_dns_cache=config.dns_cache,
            keepalive_timeout=config.keepalive_timeout,
            enable_cleanup_closed=True,
        )

        # Session pool
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        self.session_lock = asyncio.Lock()

    async def get_session(self, base_url: str) -> aiohttp.ClientSession:
        """Get or create session for base URL"""
        async with self.session_lock:
            if base_url not in self.sessions:
                timeout = aiohttp.ClientTimeout(
                    total=self.config.total_timeout,
                    connect=self.config.connect_timeout,
                    sock_read=self.config.read_timeout,
                    sock_write=self.config.write_timeout,
                )

                self.sessions[base_url] = aiohttp.ClientSession(
                    connector=self.connector,
                    timeout=timeout,
                    trust_env=True,
                )

                self.logger.info(f"Created session for {base_url}")

            return self.sessions[base_url]

    async def close_all_sessions(self):
        """Close all sessions"""
        async with self.session_lock:
            for session in self.sessions.values():
                await session.close()
            self.sessions.clear()
            self.logger.info("Closed all sessions")

    @asynccontextmanager
    async def timeout_context(self, operation_name: str = "operation"):
        """Context manager for timeout handling"""
        start_time = time.time()

        try:
            yield
        except asyncio.TimeoutError as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Timeout in {operation_name} after {elapsed:.2f}s")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Error in {operation_name} after {elapsed:.2f}s: {e}")
            raise

    def calculate_timeout(self, base_timeout: float, multiplier: float = 1.0) -> float:
        """Calculate timeout with multiplier"""
        return min(base_timeout * multiplier, self.config.total_timeout)


class RetryHandler:
    """Enhanced retry handler"""

    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.retry_counts: Dict[str, int] = {}
        self.last_retry_time: Dict[str, float] = {}

    def should_retry(
        self, exception: Exception, response_status: Optional[int] = None
    ) -> bool:
        """Check if request should be retried"""
        # Check response status
        if response_status and response_status in self.config.retry_on_status:
            return True

        # Check exception type
        if isinstance(exception, self.config.retry_on_exceptions):
            return True

        return False

    def get_delay(
        self, attempt: int, response_headers: Optional[Dict[str, str]] = None
    ) -> float:
        """Calculate delay for retry attempt"""
        # Check Retry-After header
        if (
            self.config.respect_retry_after_header
            and response_headers
            and "Retry-After" in response_headers
        ):
            retry_after = response_headers["Retry-After"]
            try:
                # Try to parse as seconds
                delay = float(retry_after)
                return min(delay, self.config.max_delay)
            except ValueError:
                # Try to parse as HTTP date
                try:
                    from email.utils import parsedate_to_datetime

                    retry_date = parsedate_to_datetime(retry_after)
                    delay = (retry_date - datetime.utcnow()).total_seconds()
                    return min(delay, self.config.max_delay)
                except (TypeError, ValueError):
                    pass

        # Calculate delay based on strategy
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_factor**attempt)
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.base_delay * self._fibonacci(attempt)
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            delay = self.config.base_delay * (self.config.backoff_factor**attempt)
            delay = self._add_jitter(delay)
        elif self.config.strategy == RetryStrategy.ADAPTIVE:
            delay = self._calculate_adaptive_delay(attempt)
        else:
            delay = self.config.base_delay

        # Clamp delay
        return max(self.config.base_delay, min(delay, self.config.max_delay))

    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1

        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b

        return b

    def _add_jitter(self, delay: float) -> float:
        """Add jitter to delay"""
        if self.config.jitter <= 0:
            return delay

        jitter_amount = delay * self.config.jitter
        jitter_range = delay * (1 + self.config.jitter)

        return random.uniform(delay - jitter_amount, jitter_range)

    def _calculate_adaptive_delay(self, attempt: int) -> float:
        """Calculate adaptive delay based on success rate"""
        # Simple adaptive strategy: increase delay if recent failures
        recent_failures = sum(
            1 for t in self.last_retry_time.values() if time.time() - t < 300
        )  # Last 5 minutes

        if recent_failures > 10:
            return self.config.base_delay * (self.config.backoff_factor**attempt) * 2
        else:
            return self.config.base_delay * (self.config.backoff_factor**attempt)

    async def retry(
        self, func: Callable, *args, operation_name: str = "operation", **kwargs
    ) -> Any:
        """Execute function with retry logic"""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                return result

            except Exception as e:
                last_exception = e

                # Check if we should retry
                if not self.should_retry(e):
                    raise

                # Check if this is the last attempt
                if attempt == self.config.max_retries:
                    self.logger.error(f"Max retries exceeded for {operation_name}")
                    raise last_exception

                # Calculate delay
                delay = self.get_delay(attempt)

                self.logger.warning(
                    f"Attempt {attempt + 1} failed for {operation_name}, "
                    f"retrying in {delay:.2f}s: {e}"
                )

                # Wait before retry
                await asyncio.sleep(delay)

        # This should never be reached, but just in case
        raise last_exception


class CircuitBreaker:
    """Enhanced circuit breaker"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
        self.next_attempt_time = 0

        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        self.total_requests += 1

        # Check circuit state
        if self.state == CircuitState.OPEN:
            if time.time() < self.next_attempt_time:
                # Call fallback function if available
                if self.config.fallback_function:
                    return await self.config.fallback_function(*args, **kwargs)
                else:
                    raise Exception("Circuit breaker is open")
            else:
                # Try to transition to half-open
                self.state = CircuitState.HALF_OPEN
                self.failure_count = 0

        try:
            result = await func(*args, **kwargs)

            # Success
            self.successful_requests += 1
            self.success_count += 1

            # Reset on success in half-open state
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info("Circuit breaker closed after successful request")

            return result

        except Exception as e:
            # Check if exception is expected
            if not isinstance(e, self.config.expected_exception):
                raise

            # Failure
            self.failed_requests += 1
            self.failure_count += 1
            self.last_failure_time = time.time()

            # Check if we should open the circuit
            if self.state == CircuitState.HALF_OPEN or (
                self.state == CircuitState.CLOSED
                and self.failure_count >= self.config.failure_threshold
            ):
                self.state = CircuitState.OPEN
                self.next_attempt_time = time.time() + self.config.recovery_timeout
                self.success_count = 0

                self.logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures. "
                    f"Will retry at {datetime.fromtimestamp(self.next_attempt_time)}"
                )

            raise e

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / self.total_requests * 100)
            if self.total_requests > 0
            else 0,
            "next_attempt_time": datetime.fromtimestamp(self.next_attempt_time)
            if self.next_attempt_time > 0
            else None,
        }


class RateLimiter:
    """Enhanced rate limiter"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Time windows
        self.second_window = []
        self.minute_window = []
        self.hour_window = []

        # Lock for thread safety
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from rate limiter"""
        async with self.lock:
            now = time.time()

            # Clean old timestamps
            self._clean_windows(now)

            # Check limits
            if (
                len(self.second_window) + tokens > self.config.burst_size
                or len(self.minute_window) + tokens > self.config.requests_per_minute
                or len(self.hour_window) + tokens > self.config.requests_per_hour
            ):
                if not self.config.dry_run:
                    return False

            # Add timestamps
            for _ in range(tokens):
                self.second_window.append(now)
                self.minute_window.append(now)
                self.hour_window.append(now)

            return True

    async def wait_if_needed(self, tokens: int = 1) -> None:
        """Wait if rate limit would be exceeded"""
        if not await self.acquire(tokens):
            # Calculate wait time
            now = time.time()

            # Find the earliest time we can proceed
            wait_time = 0

            # Check second window
            if len(self.second_window) + tokens > self.config.burst_size:
                oldest_second = min(self.second_window)
                wait_time = max(wait_time, oldest_second + 1 - now)

            # Check minute window
            if len(self.minute_window) + tokens > self.config.requests_per_minute:
                oldest_minute = min(self.minute_window)
                wait_time = max(wait_time, oldest_minute + 60 - now)

            # Check hour window
            if len(self.hour_window) + tokens > self.config.requests_per_hour:
                oldest_hour = min(self.hour_window)
                wait_time = max(wait_time, oldest_hour + 3600 - now)

            if wait_time > 0:
                self.logger.info(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                await self.acquire(tokens)

    def _clean_windows(self, now: float):
        """Clean old timestamps from windows"""
        # Clean second window (last 1 second)
        self.second_window = [t for t in self.second_window if now - t < 1]

        # Clean minute window (last 60 seconds)
        self.minute_window = [t for t in self.minute_window if now - t < 60]

        # Clean hour window (last 3600 seconds)
        self.hour_window = [t for t in self.hour_window if now - t < 3600]

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        now = time.time()

        return {
            "requests_per_second": self.config.requests_per_second,
            "requests_per_minute": self.config.requests_per_minute,
            "requests_per_hour": self.config.requests_per_hour,
            "burst_size": self.config.burst_size,
            "current_second_count": len([t for t in self.second_window if now - t < 1]),
            "current_minute_count": len(
                [t for t in self.minute_window if now - t < 60]
            ),
            "current_hour_count": len([t for t in self.hour_window if now - t < 3600]),
            "dry_run": self.config.dry_run,
        }


class ResilienceManager:
    """Main resilience manager combining all handlers"""

    def __init__(
        self,
        timeout_config: Optional[TimeoutConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
    ):
        self.timeout_config = timeout_config or TimeoutConfig()
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.rate_limit_config = rate_limit_config or RateLimitConfig()

        # Initialize handlers
        self.timeout_handler = TimeoutHandler(self.timeout_config)
        self.retry_handler = RetryHandler(self.retry_config)
        self.circuit_breaker = CircuitBreaker(self.circuit_config)
        self.rate_limiter = RateLimiter(self.rate_limit_config)

        self.logger = logging.getLogger(self.__class__.__name__)

    async def execute(
        self,
        func: Callable,
        *args,
        operation_name: str = "operation",
        tokens: int = 1,
        **kwargs,
    ) -> Any:
        """Execute function with full resilience protection"""

        # Rate limiting
        await self.rate_limiter.wait_if_needed(tokens)

        # Circuit breaker
        wrapped_func = functools.partial(
            self._execute_with_retry, func, operation_name, *args, **kwargs
        )

        return await self.circuit_breaker.call(wrapped_func)

    async def _execute_with_retry(
        self, func: Callable, operation_name: str, *args, **kwargs
    ) -> Any:
        """Execute function with retry logic"""
        return await self.retry_handler.retry(
            func, *args, operation_name=operation_name, **kwargs
        )

    async def execute_with_timeout(
        self,
        func: Callable,
        *args,
        operation_name: str = "operation",
        tokens: int = 1,
        **kwargs,
    ) -> Any:
        """Execute function with timeout protection"""

        async with self.timeout_handler.timeout_context(operation_name):
            return await self.execute(
                func, *args, operation_name=operation_name, tokens=tokens, **kwargs
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics"""
        return {
            "timeout": {
                "connect_timeout": self.timeout_config.connect_timeout,
                "read_timeout": self.timeout_config.read_timeout,
                "total_timeout": self.timeout_config.total_timeout,
            },
            "retry": {
                "max_retries": self.retry_config.max_retries,
                "strategy": self.retry_config.strategy.value,
            },
            "circuit_breaker": self.circuit_breaker.get_state(),
            "rate_limiter": self.rate_limiter.get_stats(),
        }

    async def close(self):
        """Close all resources"""
        await self.timeout_handler.close_all_sessions()
        self.logger.info("Resilience manager closed")


# Global resilience manager instance
_resilience_manager: Optional[ResilienceManager] = None


def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager instance"""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager


def set_resilience_manager(manager: ResilienceManager):
    """Set global resilience manager instance"""
    global _resilience_manager
    _resilience_manager = manager


# Decorators for easy usage
def resilient(
    max_retries: int = 3,
    timeout: float = 30.0,
    operation_name: Optional[str] = None,
    tokens: int = 1,
):
    """Decorator for resilient function execution"""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_resilience_manager()

            # Configure if needed
            manager.retry_config.max_retries = max_retries
            manager.timeout_config.total_timeout = timeout

            op_name = operation_name or func.__name__
            return await manager.execute_with_timeout(
                func, *args, operation_name=op_name, tokens=tokens, **kwargs
            )

        return wrapper

    return decorator


def with_circuit_breaker(fallback_function: Optional[Callable] = None):
    """Decorator for circuit breaker protection"""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_resilience_manager()

            if fallback_function:
                manager.circuit_config.fallback_function = fallback_function

            return await manager.execute(
                func, *args, operation_name=func.__name__, **kwargs
            )

        return wrapper

    return decorator


def rate_limited(tokens: int = 1):
    """Decorator for rate limiting"""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            return await manager.execute(
                func, *args, operation_name=func.__name__, tokens=tokens, **kwargs
            )

        return wrapper

    return decorator
