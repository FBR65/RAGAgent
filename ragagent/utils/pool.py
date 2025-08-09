"""
Request pooling and connection management for the Agentic RAG system
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import weakref
from contextlib import asynccontextmanager
import aiohttp
import httpx

from ..models import ProcessingRequest, ProcessingResponse, APIResponse

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Priority levels for requests"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class PooledRequest:
    """Represents a request in the pool"""

    request: ProcessingRequest
    future: asyncio.Future
    priority: RequestPriority = RequestPriority.NORMAL
    created_at: float = None
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class ConnectionPool:
    """Connection pool for HTTP clients"""

    def __init__(self, max_connections: int = 10, max_keepalive_connections: int = 5):
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self._pool = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._active_connections = 0
        self._total_created = 0

        # Pre-warm the pool
        self._prewarm()

    def _prewarm(self):
        """Pre-warm the connection pool with initial connections"""
        try:
            for _ in range(
                min(self.max_keepalive_connections, self.max_connections // 2)
            ):
                client = self._create_client()
                if client:
                    self._pool.put(client)
        except Exception as e:
            logger.warning(f"Failed to pre-warm connection pool: {e}")

    def _create_client(self) -> Optional[httpx.AsyncClient]:
        """Create a new HTTP client"""
        try:
            client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_keepalive_connections,
                ),
                timeout=httpx.Timeout(30.0),
            )
            self._total_created += 1
            return client
        except Exception as e:
            logger.error(f"Failed to create HTTP client: {e}")
            return None

    async def get_client(self) -> httpx.AsyncClient:
        """Get a client from the pool"""
        start_time = time.time()

        try:
            # Try to get a client from the pool with timeout
            try:
                client = self._pool.get(timeout=5.0)
                logger.debug(
                    f"Got client from pool (waited {time.time() - start_time:.2f}s)"
                )
                return client
            except queue.Empty:
                # Pool is empty, create a new client if we haven't reached the limit
                with self._lock:
                    if self._active_connections < self.max_connections:
                        self._active_connections += 1
                        logger.debug("Created new client (pool limit not reached)")
                        return self._create_client()
                    else:
                        # Wait for a client to be returned
                        logger.warning("Connection pool exhausted, waiting...")
                        client = self._pool.get(timeout=30.0)
                        return client
        except Exception as e:
            logger.error(f"Failed to get client from pool: {e}")
            # Fallback to creating a new client
            return self._create_client()

    async def return_client(self, client: httpx.AsyncClient):
        """Return a client to the pool"""
        try:
            # Check if client is still healthy
            if client.is_closed:
                logger.debug("Client is closed, not returning to pool")
                with self._lock:
                    self._active_connections -= 1
                return

            # Reset client state
            client.timeout = httpx.Timeout(30.0)

            # Return to pool if space available
            try:
                self._pool.put_nowait(client)
                logger.debug("Returned client to pool")
            except queue.Full:
                logger.debug("Pool is full, closing client")
                await client.aclose()
                with self._lock:
                    self._active_connections -= 1

        except Exception as e:
            logger.error(f"Failed to return client to pool: {e}")
            try:
                await client.aclose()
            except:
                pass
            with self._lock:
                self._active_connections -= 1

    async def close_all(self):
        """Close all clients in the pool"""
        while not self._pool.empty():
            try:
                client = self._pool.get_nowait()
                await client.aclose()
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error closing client: {e}")

        with self._lock:
            self._active_connections = 0

        logger.info("All connection pool clients closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            "pool_size": self._pool.qsize(),
            "active_connections": self._active_connections,
            "max_connections": self.max_connections,
            "total_created": self._total_created,
        }


class RequestPool:
    """Manages request prioritization and execution"""

    def __init__(
        self,
        max_concurrent_requests: int = 5,
        request_timeout: float = 300.0,
        enable_priority: bool = True,
    ):
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        self.enable_priority = enable_priority

        # Priority queues for different request types
        self._queues = {
            RequestPriority.CRITICAL: queue.PriorityQueue(),
            RequestPriority.HIGH: queue.PriorityQueue(),
            RequestPriority.NORMAL: queue.PriorityQueue(),
            RequestPriority.LOW: queue.PriorityQueue(),
        }

        self._active_requests = 0
        self._lock = threading.Lock()
        self._shutdown = False

        # Worker threads
        self._workers = []
        self._start_workers()

    def _start_workers(self):
        """Start worker threads for processing requests"""
        for i in range(self.max_concurrent_requests):
            worker = threading.Thread(
                target=self._worker_loop, name=f"RequestPoolWorker-{i}", daemon=True
            )
            worker.start()
            self._workers.append(worker)

        logger.info(f"Started {self.max_concurrent_requests} request pool workers")

    def _worker_loop(self):
        """Main worker loop for processing requests"""
        while not self._shutdown:
            try:
                # Get next request from any priority queue
                request = self._get_next_request()
                if request is None:
                    time.sleep(0.1)
                    continue

                # Process the request
                self._process_request(request)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1.0)

    def _get_next_request(self) -> Optional[PooledRequest]:
        """Get the next request from priority queues"""
        # Check queues in priority order
        for priority in sorted(RequestPriority, key=lambda p: p.value, reverse=True):
            if self._queues[priority].empty():
                continue

            try:
                # Get request with timeout
                priority_value, request = self._queues[priority].get(timeout=0.1)
                return request
            except queue.Empty:
                continue

        return None

    def _process_request(self, pooled_request: PooledRequest):
        """Process a single request"""
        start_time = time.time()

        try:
            with self._lock:
                self._active_requests += 1

            logger.debug(f"Processing request (priority: {pooled_request.priority})")

            # Here you would implement the actual request processing
            # For now, we'll simulate it
            result = self._execute_request(pooled_request.request)

            # Set the result in the future
            if not pooled_request.future.done():
                pooled_request.future.set_result(result)

            processing_time = time.time() - start_time
            logger.debug(f"Request completed in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Error processing request: {e}")

            # Handle retries
            if pooled_request.retry_count < pooled_request.max_retries:
                pooled_request.retry_count += 1
                logger.info(f"Retrying request (attempt {pooled_request.retry_count})")

                # Schedule retry with exponential backoff
                delay = min(2**pooled_request.retry_count, 30)  # Max 30 seconds
                threading.Timer(
                    delay, self._retry_request, args=[pooled_request]
                ).start()
            else:
                # Set exception in future
                if not pooled_request.future.done():
                    pooled_request.future.set_exception(e)

        finally:
            with self._lock:
                self._active_requests -= 1

    def _execute_request(self, request: ProcessingRequest) -> APIResponse:
        """Execute a single request (placeholder implementation)"""
        # This would be replaced with actual request processing
        # For now, return a dummy response
        from ..models import ProcessingResponse, ErrorDetail

        return APIResponse(
            success=True,
            data=ProcessingResponse(
                question=request.question,
                answer="This is a placeholder response",
                citations=[],
                confidence_score=0.5,
                source_paragraphs=0,
                total_processing_steps=0,
                processing_time=0.0,
                metadata={},
            ),
        )

    def _retry_request(self, pooled_request: PooledRequest):
        """Retry a failed request"""
        try:
            self._queues[pooled_request.priority].put((time.time(), pooled_request))
        except Exception as e:
            logger.error(f"Failed to schedule retry: {e}")

    def submit_request(
        self,
        request: ProcessingRequest,
        priority: RequestPriority = RequestPriority.NORMAL,
        callback: Optional[Callable[[APIResponse], None]] = None,
    ) -> asyncio.Future:
        """Submit a request to the pool"""

        # Create future for async result
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        future = loop.create_future()

        # Create pooled request
        pooled_request = PooledRequest(
            request=request, future=future, priority=priority, max_retries=3
        )

        # Add to appropriate queue
        try:
            self._queues[priority].put((time.time(), pooled_request))
            logger.debug(f"Request submitted with priority {priority}")
        except Exception as e:
            logger.error(f"Failed to submit request: {e}")
            if not future.done():
                future.set_exception(e)

        return future

    def get_stats(self) -> Dict[str, Any]:
        """Get request pool statistics"""
        queue_sizes = {priority.name: q.qsize() for priority, q in self._queues.items()}

        return {
            "active_requests": self._active_requests,
            "max_concurrent_requests": self.max_concurrent_requests,
            "queue_sizes": queue_sizes,
            "total_queued": sum(q.qsize() for q in self._queues.values()),
            "enable_priority": self.enable_priority,
        }

    def shutdown(self):
        """Shutdown the request pool"""
        self._shutdown = True

        # Cancel all pending requests
        for priority_queue in self._queues.values():
            while not priority_queue.empty():
                try:
                    _, pooled_request = priority_queue.get_nowait()
                    if not pooled_request.future.done():
                        pooled_request.future.cancel()
                except queue.Empty:
                    break

        logger.info("Request pool shutdown complete")


class RequestPoolManager:
    """Manages connection and request pools"""

    def __init__(
        self,
        max_connections: int = 10,
        max_concurrent_requests: int = 5,
        enable_priority: bool = True,
    ):
        self.connection_pool = ConnectionPool(max_connections=max_connections)
        self.request_pool = RequestPool(
            max_concurrent_requests=max_concurrent_requests,
            enable_priority=enable_priority,
        )

        # Track active requests
        self._active_requests = weakref.WeakSet()

    async def process_request(
        self,
        request: ProcessingRequest,
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> APIResponse:
        """Process a request through the pool system"""

        # Create future for the result
        future = self.request_pool.submit_request(request, priority)

        # Track the request
        self._active_requests.add(request)

        try:
            # Wait for result with timeout
            result = await asyncio.wait_for(
                future, timeout=self.request_pool.request_timeout
            )
            return result

        except asyncio.TimeoutError:
            logger.error(
                f"Request timed out after {self.request_pool.request_timeout}s"
            )
            raise TimeoutError(
                f"Request timed out after {self.request_pool.request_timeout} seconds"
            )

        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            raise

        finally:
            self._active_requests.discard(request)

    def get_stats(self) -> Dict[str, Any]:
        """Get combined pool statistics"""
        return {
            "connection_pool": self.connection_pool.get_stats(),
            "request_pool": self.request_pool.get_stats(),
            "active_requests": len(self._active_requests),
        }

    async def close(self):
        """Close all pools"""
        await self.connection_pool.close_all()
        self.request_pool.shutdown()
        logger.info("Request pool manager closed")


# Context manager for request pooling
@asynccontextmanager
async def pooled_request_processing(pool_manager: RequestPoolManager):
    """Context manager for pooled request processing"""
    try:
        yield pool_manager
    finally:
        # Cleanup is handled by the manager's close method
        pass
