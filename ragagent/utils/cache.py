"""
Caching mechanisms for the Agentic RAG system
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import pickle
import threading
from functools import wraps

from ..models import ProcessingRequest, ProcessingResponse, Chunk

logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a single cache entry with metadata"""

    def __init__(self, key: str, value: Any, ttl: int = 3600):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if the cache entry has expired"""
        return time.time() - self.created_at > self.ttl

    def touch(self):
        """Update last access time and increment access count"""
        self.last_accessed = time.time()
        self.access_count += 1


class MemoryCache:
    """In-memory cache with LRU eviction policy"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def _generate_key(self, request: ProcessingRequest) -> str:
        """Generate a unique key for the processing request"""
        # Create a deterministic string representation of the request
        key_data = {
            "question": request.question,
            "document_path": str(request.document_path),
            "document_type": request.document_type.value
            if request.document_type
            else None,
            "config": request.config.dict() if request.config else None,
        }
        key_string = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, request: ProcessingRequest) -> Optional[ProcessingResponse]:
        """Get cached response for a processing request"""
        with self.lock:
            key = self._generate_key(request)

            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    entry.touch()
                    self.hits += 1
                    logger.debug(f"Cache hit for key: {key[:8]}...")
                    return entry.value
                else:
                    # Remove expired entry
                    del self.cache[key]
                    logger.debug(f"Cache expired for key: {key[:8]}...")

            self.misses += 1
            logger.debug(f"Cache miss for key: {key[:8]}...")
            return None

    def set(
        self,
        request: ProcessingRequest,
        response: ProcessingResponse,
        ttl: Optional[int] = None,
    ):
        """Cache a processing response"""
        with self.lock:
            key = self._generate_key(request)
            entry = CacheEntry(key, response, ttl or self.default_ttl)

            # Evict oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_oldest()

            self.cache[key] = entry
            logger.debug(f"Cached response for key: {key[:8]}...")

    def _evict_oldest(self):
        """Evict the least recently used entries"""
        if not self.cache:
            return

        # Sort by last access time
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].last_accessed)

        # Remove oldest 10% of entries
        evict_count = max(1, len(self.cache) // 10)
        for key, _ in sorted_entries[:evict_count]:
            del self.cache[key]

        logger.debug(f"Evicted {evict_count} cache entries")

    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "size": len(self.cache),
                "max_size": self.max_size,
                "total_requests": total_requests,
            }


class DiskCache:
    """Disk-based cache for persistent storage"""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size_mb: int = 1000,
        default_ttl: int = 3600,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.lock = threading.RLock()

        # Background cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._background_cleanup, daemon=True
        )
        self._cleanup_thread.start()

    def _get_file_path(self, key: str) -> Path:
        """Get file path for a cache key"""
        return self.cache_dir / f"{key}.cache"

    def get(self, request: ProcessingRequest) -> Optional[ProcessingResponse]:
        """Get cached response from disk"""
        with self.lock:
            key = self._generate_key(request)
            file_path = self._get_file_path(key)

            if not file_path.exists():
                return None

            try:
                with open(file_path, "rb") as f:
                    entry = pickle.load(f)

                if entry.is_expired():
                    file_path.unlink()
                    return None

                entry.touch()
                self._update_metadata(key, entry)
                logger.debug(f"Disk cache hit for key: {key[:8]}...")
                return entry.value

            except Exception as e:
                logger.error(f"Error reading disk cache: {e}")
                if file_path.exists():
                    file_path.unlink()
                return None

    def set(
        self,
        request: ProcessingRequest,
        response: ProcessingResponse,
        ttl: Optional[int] = None,
    ):
        """Cache response to disk"""
        with self.lock:
            key = self._generate_key(request)
            entry = CacheEntry(key, response, ttl or self.default_ttl)

            try:
                file_path = self._get_file_path(key)

                # Write to temporary file first
                temp_path = file_path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    pickle.dump(entry, f)

                # Atomic move
                temp_path.rename(file_path)

                # Check cache size and cleanup if needed
                self._check_size()

                logger.debug(f"Disk cached response for key: {key[:8]}...")

            except Exception as e:
                logger.error(f"Error writing disk cache: {e}")

    def _generate_key(self, request: ProcessingRequest) -> str:
        """Generate a unique key for the processing request"""
        key_data = {
            "question": request.question,
            "document_path": str(request.document_path),
            "document_type": request.document_type.value
            if request.document_type
            else None,
            "config": request.config.dict() if request.config else None,
        }
        key_string = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _update_metadata(self, key: str, entry: CacheEntry):
        """Update metadata for cache entry"""
        # This would store metadata separately in a real implementation
        pass

    def _check_size(self):
        """Check cache size and cleanup if needed"""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache"))

            if total_size > self.max_size_bytes:
                self._cleanup_by_size()

        except Exception as e:
            logger.error(f"Error checking cache size: {e}")

    def _cleanup_by_size(self):
        """Cleanup cache entries by size (LRU)"""
        try:
            files = []
            for file_path in self.cache_dir.glob("*.cache"):
                stat = file_path.stat()
                files.append((file_path, stat.st_mtime, stat.st_size))

            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x[1])

            # Remove oldest files until we're under the size limit
            total_size = sum(f[2] for f in files)
            target_size = self.max_size_bytes * 0.8  # Clean up to 80% of limit

            removed_count = 0
            for file_path, _, size in files:
                if total_size <= target_size:
                    break

                file_path.unlink()
                total_size -= size
                removed_count += 1

            logger.info(f"Removed {removed_count} cache files to free space")

        except Exception as e:
            logger.error(f"Error during size-based cleanup: {e}")

    def _background_cleanup(self):
        """Background cleanup thread"""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")

    def _cleanup_expired(self):
        """Remove expired cache entries"""
        try:
            current_time = time.time()
            removed_count = 0

            for file_path in self.cache_dir.glob("*.cache"):
                try:
                    with open(file_path, "rb") as f:
                        entry = pickle.load(f)

                    if entry.is_expired():
                        file_path.unlink()
                        removed_count += 1

                except Exception as e:
                    logger.warning(f"Error checking cache file {file_path}: {e}")
                    file_path.unlink()  # Remove corrupted files

            if removed_count > 0:
                logger.info(f"Removed {removed_count} expired cache files")

        except Exception as e:
            logger.error(f"Error during expired cleanup: {e}")

    def clear(self):
        """Clear all cache files"""
        with self.lock:
            for file_path in self.cache_dir.glob("*.cache"):
                file_path.unlink()
            logger.info("Disk cache cleared")


class CacheManager:
    """Unified cache manager combining memory and disk cache"""

    def __init__(
        self,
        memory_cache_size: int = 1000,
        disk_cache_dir: Optional[Union[str, Path]] = None,
        disk_cache_size_mb: int = 1000,
        default_ttl: int = 3600,
    ):
        self.memory_cache = MemoryCache(
            max_size=memory_cache_size, default_ttl=default_ttl
        )

        if disk_cache_dir:
            self.disk_cache = DiskCache(
                cache_dir=disk_cache_dir,
                max_size_mb=disk_cache_size_mb,
                default_ttl=default_ttl,
            )
        else:
            self.disk_cache = None

        self.default_ttl = default_ttl

    def get(self, request: ProcessingRequest) -> Optional[ProcessingResponse]:
        """Get cached response, checking memory cache first, then disk cache"""
        # Try memory cache first
        response = self.memory_cache.get(request)
        if response:
            return response

        # Try disk cache if available
        if self.disk_cache:
            response = self.disk_cache.get(request)
            if response:
                # Update memory cache with disk hit
                self.memory_cache.set(request, response, self.default_ttl)
                return response

        return None

    def set(
        self,
        request: ProcessingRequest,
        response: ProcessingResponse,
        ttl: Optional[int] = None,
    ):
        """Cache response in both memory and disk cache"""
        # Set in memory cache
        self.memory_cache.set(request, response, ttl)

        # Set in disk cache if available
        if self.disk_cache:
            self.disk_cache.set(request, response, ttl)

    def clear(self):
        """Clear all caches"""
        self.memory_cache.clear()
        if self.disk_cache:
            self.disk_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics"""
        stats = self.memory_cache.get_stats()

        if self.disk_cache:
            # Add disk cache stats
            stats["disk_cache_enabled"] = True
            # Could add disk cache stats here if implemented
        else:
            stats["disk_cache_enabled"] = False

        return stats


# Decorator for caching function results
def cache_results(cache_manager: CacheManager, ttl: Optional[int] = None):
    """Decorator to cache function results"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a request key from function arguments
            # This is a simplified version - in practice, you'd need more sophisticated key generation
            request_key = {
                "func": func.__name__,
                "args": str(args),
                "kwargs": str(sorted(kwargs.items())),
            }

            # For demonstration, we'll create a dummy ProcessingRequest
            # In a real implementation, you'd need to extract the actual request
            try:
                # Try to extract ProcessingRequest from arguments
                if args and isinstance(args[0], ProcessingRequest):
                    request = args[0]
                    cached_result = cache_manager.get(request)
                    if cached_result is not None:
                        logger.debug(f"Returning cached result for {func.__name__}")
                        return cached_result
            except (IndexError, TypeError):
                pass

            # Execute function
            result = func(*args, **kwargs)

            # Cache the result if it's a ProcessingResponse
            if (
                args
                and isinstance(args[0], ProcessingRequest)
                and isinstance(result, ProcessingResponse)
            ):
                cache_manager.set(args[0], result, ttl)

            return result

        return wrapper

    return decorator
