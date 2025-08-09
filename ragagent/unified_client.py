"""
Unified client for OpenAI-compatible APIs
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
from contextlib import asynccontextmanager

from .models import (
    ProcessingRequest,
    ProcessingResponse,
    Chunk,
    AnswerWithCitations,
    VerificationResult,
    ErrorDetail,
    APIResponse,
    DocumentType,
)

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported AI providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    OLLAMA = "ollama"
    LOCAL = "local"
    AZURE = "azure"
    GOOGLE = "google"


@dataclass
class ClientConfig:
    """Configuration for the unified client"""

    provider: ProviderType = ProviderType.OPENAI
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 30
    retry_count: int = 3
    retry_delay: float = 1.0
    max_retries: int = 5
    request_timeout: int = 60
    connect_timeout: int = 10
    read_timeout: int = 30
    verify_ssl: bool = True
    proxy: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    # Azure specific
    azure_deployment: Optional[str] = None
    azure_api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    # Anthropic specific
    anthropic_version: Optional[str] = None
    # Google specific
    google_project: Optional[str] = None
    google_location: Optional[str] = None
    google_credentials: Optional[str] = None


class RateLimiter:
    """Rate limiter for API requests"""

    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 90000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_times = []
        self.token_times = []

    async def wait_if_needed(self, tokens: int = 1):
        """Wait if rate limit would be exceeded"""
        now = time.time()

        # Clean old timestamps
        self.request_times = [t for t in self.request_times if now - t < 60]
        self.token_times = [t for t in self.token_times if now - t < 60]

        # Check request limit
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        # Check token limit
        if len(self.token_times) + tokens > self.tokens_per_minute:
            sleep_time = 60 - (now - self.token_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        # Record request
        self.request_times.append(now)
        self.token_times.append(now)


class CircuitBreaker:
    """Circuit breaker for API requests"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open

    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"

            raise e


class RetryHandler:
    """Retry handler for failed requests"""

    def __init__(
        self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def retry(self, func, *args, **kwargs):
        """Retry function with exponential backoff"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2**attempt), self.max_delay)
                    await asyncio.sleep(delay)
                else:
                    raise last_exception


class UnifiedClient:
    """Unified client for OpenAI-compatible APIs"""

    def __init__(self, config: ClientConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.rate_limiter = RateLimiter()
        self.circuit_breaker = CircuitBreaker()
        self.retry_handler = RetryHandler(config.max_retries, config.retry_delay)

        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None

        # Provider-specific configurations
        self._setup_provider_config()

        self.logger.info(f"Unified client initialized for {config.provider.value}")

    def _setup_provider_config(self):
        """Setup provider-specific configuration"""
        if self.config.provider == ProviderType.OPENAI:
            self.config.base_url = self.config.base_url or "https://api.openai.com/v1"
            self.config.headers = self.config.headers or {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            }
            if self.config.organization:
                self.config.headers["OpenAI-Organization"] = self.config.organization
            if self.config.project:
                self.config.headers["OpenAI-Project"] = self.config.project

        elif self.config.provider == ProviderType.AZURE:
            if not self.config.azure_deployment:
                raise ValueError("Azure deployment name is required")
            if not self.config.azure_api_version:
                raise ValueError("Azure API version is required")
            if not self.config.azure_endpoint:
                raise ValueError("Azure endpoint is required")

            self.config.base_url = f"{self.config.azure_endpoint}/openai/deployments/{self.config.azure_deployment}"
            self.config.headers = self.config.headers or {
                "Content-Type": "application/json",
                "api-key": self.config.api_key,
            }

        elif self.config.provider == ProviderType.ANTHROPIC:
            self.config.base_url = self.config.base_url or "https://api.anthropic.com"
            self.config.headers = self.config.headers or {
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key,
                "anthropic-version": self.config.anthropic_version or "2023-06-01",
            }

        elif self.config.provider == ProviderType.GOOGLE:
            self.config.base_url = (
                self.config.base_url
                or "https://generativelanguage.googleapis.com/v1beta"
            )
            self.config.headers = self.config.headers or {
                "Content-Type": "application/json",
            }

        elif self.config.provider == ProviderType.OLLAMA:
            self.config.base_url = self.config.base_url or "http://localhost:11434"
            self.config.headers = self.config.headers or {
                "Content-Type": "application/json",
            }

        elif self.config.provider == ProviderType.COHERE:
            self.config.base_url = self.config.base_url or "https://api.cohere.com/v1"
            self.config.headers = self.config.headers or {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            }

        elif self.config.provider == ProviderType.LOCAL:
            self.config.base_url = self.config.base_url or "http://localhost:8000"
            self.config.headers = self.config.headers or {
                "Content-Type": "application/json",
            }

    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.config.request_timeout,
                connect=self.config.connect_timeout,
                sock_read=self.config.read_timeout,
            )

            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.config.headers,
                trust_env=True,  # Use proxy from environment
            )

    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Make HTTP request with error handling"""
        await self._ensure_session()

        url = f"{self.config.base_url}{endpoint}"

        # Prepare request data
        request_data = json.dumps(data) if data else None

        # Rate limiting
        await self.rate_limiter.wait_if_needed()

        # Circuit breaker
        wrapped_request = self.circuit_breaker.call(
            self._execute_request, method, url, request_data, stream, **kwargs
        )

        # Retry handler
        return await self.retry_handler.retry(wrapped_request, **kwargs)

    async def _execute_request(
        self, method: str, url: str, data: Optional[str], stream: bool, **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Execute HTTP request"""
        try:
            async with self.session.request(
                method,
                url,
                data=data,
                proxy=self.config.proxy,
                ssl=self.config.verify_ssl,
                **kwargs,
            ) as response:
                if stream:
                    return self._stream_response(response)
                else:
                    return await self._handle_response(response)

        except asyncio.TimeoutError:
            raise Exception(
                f"Request timeout after {self.config.request_timeout} seconds"
            )
        except aiohttp.ClientError as e:
            raise Exception(f"HTTP request failed: {e}")

    async def _handle_response(
        self, response: aiohttp.ClientResponse
    ) -> Dict[str, Any]:
        """Handle HTTP response"""
        try:
            response_data = await response.json()
        except json.JSONDecodeError:
            response_text = await response.text()
            raise Exception(f"Invalid JSON response: {response_text}")

        if response.status >= 400:
            error_msg = response_data.get("error", {}).get(
                "message", str(response_data)
            )
            raise Exception(f"API Error {response.status}: {error_msg}")

        return response_data

    def _stream_response(
        self, response: aiohttp.ClientResponse
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream HTTP response"""
        buffer = ""

        async def _stream_generator():
            nonlocal buffer

            async for chunk in response.content:
                buffer += chunk.decode()

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            return

                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue

        return _stream_generator()

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Create chat completion"""
        model = model or self.config.model
        temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        # Prepare request data based on provider
        if self.config.provider == ProviderType.OPENAI:
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }

            if tools:
                data["tools"] = tools
            if tool_choice:
                data["tool_choice"] = tool_choice
            if response_format:
                data["response_format"] = response_format

        elif self.config.provider == ProviderType.ANTHROPIC:
            data = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": stream,
            }

            if temperature is not None:
                data["temperature"] = temperature

        elif self.config.provider == ProviderType.GOOGLE:
            data = {
                "contents": messages,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
                "safetySettings": kwargs.get("safety_settings", []),
            }

        elif self.config.provider == ProviderType.OLLAMA:
            data = {
                "model": model,
                "prompt": self._format_messages_for_ollama(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }

        elif self.config.provider == ProviderType.COHERE:
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }

        else:
            raise ValueError(f"Provider {self.config.provider} not supported")

        # Make request
        endpoint = self._get_chat_endpoint()
        return await self._make_request("POST", endpoint, data, stream)

    def _get_chat_endpoint(self) -> str:
        """Get chat completion endpoint for provider"""
        if self.config.provider == ProviderType.OPENAI:
            return "/chat/completions"
        elif self.config.provider == ProviderType.ANTHROPIC:
            return "/messages"
        elif self.config.provider == ProviderType.GOOGLE:
            return f"/models/{self.config.model}:generateContent"
        elif self.config.provider == ProviderType.OLLAMA:
            return "/api/generate"
        elif self.config.provider == ProviderType.COHERE:
            return "/chat"
        elif self.config.provider == ProviderType.LOCAL:
            return "/v1/chat/completions"
        else:
            raise ValueError(f"Provider {self.config.provider} not supported")

    def _format_messages_for_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Ollama"""
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n\n"
        return prompt

    async def embeddings(
        self, input: Union[str, List[str]], model: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Create embeddings"""
        model = model or self.config.model

        # Prepare request data based on provider
        if self.config.provider == ProviderType.OPENAI:
            data = {
                "model": model,
                "input": input,
            }

        elif self.config.provider == ProviderType.ANTHROPIC:
            raise ValueError("Anthropic does not support embeddings")

        elif self.config.provider == ProviderType.GOOGLE:
            data = {
                "model": model,
                "content": input,
            }

        elif self.config.provider == ProviderType.OLLAMA:
            data = {
                "model": model,
                "prompt": input if isinstance(input, str) else input[0],
            }

        elif self.config.provider == ProviderType.COHERE:
            data = {
                "model": model,
                "texts": input if isinstance(input, list) else [input],
            }

        else:
            raise ValueError(f"Provider {self.config.provider} not supported")

        # Make request
        endpoint = self._get_embeddings_endpoint()
        return await self._make_request("POST", endpoint, data)

    def _get_embeddings_endpoint(self) -> str:
        """Get embeddings endpoint for provider"""
        if self.config.provider == ProviderType.OPENAI:
            return "/embeddings"
        elif self.config.provider == ProviderType.GOOGLE:
            return f"/models/{self.config.model}:embedText"
        elif self.config.provider == ProviderType.OLLAMA:
            return "/api/embeddings"
        elif self.config.provider == ProviderType.COHERE:
            return "/embed"
        elif self.config.provider == ProviderType.LOCAL:
            return "/v1/embeddings"
        else:
            raise ValueError(f"Provider {self.config.provider} not supported")

    async def models(self) -> Dict[str, Any]:
        """List available models"""
        if self.config.provider == ProviderType.OPENAI:
            return await self._make_request("GET", "/models")
        elif self.config.provider == ProviderType.ANTHROPIC:
            return await self._make_request("GET", "/models")
        elif self.config.provider == ProviderType.GOOGLE:
            return await self._make_request("GET", "/models")
        elif self.config.provider == ProviderType.OLLAMA:
            return await self._make_request("GET", "/api/tags")
        elif self.config.provider == ProviderType.COHERE:
            return await self._make_request("GET", "/models")
        elif self.config.provider == ProviderType.LOCAL:
            return await self._make_request("GET", "/v1/models")
        else:
            raise ValueError(f"Provider {self.config.provider} not supported")

    async def health_check(self) -> bool:
        """Check API health"""
        try:
            if self.config.provider == ProviderType.OLLAMA:
                return await self._make_request("GET", "/api/tags")
            elif self.config.provider == ProviderType.LOCAL:
                return await self._make_request("GET", "/v1/models")
            else:
                # For most APIs, we'll just try to list models
                await self.models()
                return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


class ClientFactory:
    """Factory for creating unified clients"""

    @staticmethod
    def create_openai_client(
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        **kwargs,
    ) -> UnifiedClient:
        """Create OpenAI client"""
        config = ClientConfig(
            provider=ProviderType.OPENAI,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return UnifiedClient(config)

    @staticmethod
    def create_azure_client(
        api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        azure_api_version: str,
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        **kwargs,
    ) -> UnifiedClient:
        """Create Azure OpenAI client"""
        config = ClientConfig(
            provider=ProviderType.AZURE,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            azure_api_version=azure_api_version,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return UnifiedClient(config)

    @staticmethod
    def create_anthropic_client(
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        anthropic_version: Optional[str] = None,
        **kwargs,
    ) -> UnifiedClient:
        """Create Anthropic client"""
        config = ClientConfig(
            provider=ProviderType.ANTHROPIC,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_version=anthropic_version,
            **kwargs,
        )
        return UnifiedClient(config)

    @staticmethod
    def create_google_client(
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gemini-pro",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        google_project: Optional[str] = None,
        google_location: Optional[str] = None,
        google_credentials: Optional[str] = None,
        **kwargs,
    ) -> UnifiedClient:
        """Create Google client"""
        config = ClientConfig(
            provider=ProviderType.GOOGLE,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
            **kwargs,
        )
        return UnifiedClient(config)

    @staticmethod
    def create_ollama_client(
        base_url: Optional[str] = None,
        model: str = "llama2",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        **kwargs,
    ) -> UnifiedClient:
        """Create Ollama client"""
        config = ClientConfig(
            provider=ProviderType.OLLAMA,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return UnifiedClient(config)

    @staticmethod
    def create_cohere_client(
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "command-r-plus",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        **kwargs,
    ) -> UnifiedClient:
        """Create Cohere client"""
        config = ClientConfig(
            provider=ProviderType.COHERE,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return UnifiedClient(config)

    @staticmethod
    def create_local_client(
        base_url: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        **kwargs,
    ) -> UnifiedClient:
        """Create local API client"""
        config = ClientConfig(
            provider=ProviderType.LOCAL,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return UnifiedClient(config)

    @staticmethod
    def create_from_env() -> UnifiedClient:
        """Create client from environment variables"""
        import os

        provider = os.getenv("AI_PROVIDER", "openai").lower()
        api_key = os.getenv("AI_API_KEY")

        if not api_key and provider != "ollama" and provider != "local":
            raise ValueError(f"API key required for provider: {provider}")

        if provider == "openai":
            return ClientFactory.create_openai_client(
                api_key=api_key,
                base_url=os.getenv("AI_BASE_URL"),
                model=os.getenv("AI_MODEL", "gpt-4"),
                temperature=float(os.getenv("AI_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("AI_MAX_TOKENS", "4000")),
            )
        elif provider == "azure":
            return ClientFactory.create_azure_client(
                api_key=api_key,
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                azure_api_version=os.getenv("AZURE_API_VERSION"),
                model=os.getenv("AI_MODEL", "gpt-4"),
                temperature=float(os.getenv("AI_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("AI_MAX_TOKENS", "4000")),
            )
        elif provider == "anthropic":
            return ClientFactory.create_anthropic_client(
                api_key=api_key,
                base_url=os.getenv("AI_BASE_URL"),
                model=os.getenv("AI_MODEL", "claude-3-sonnet-20240229"),
                temperature=float(os.getenv("AI_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("AI_MAX_TOKENS", "4000")),
                anthropic_version=os.getenv("ANTHROPIC_VERSION"),
            )
        elif provider == "google":
            return ClientFactory.create_google_client(
                api_key=api_key,
                base_url=os.getenv("AI_BASE_URL"),
                model=os.getenv("AI_MODEL", "gemini-pro"),
                temperature=float(os.getenv("AI_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("AI_MAX_TOKENS", "4000")),
                google_project=os.getenv("GOOGLE_PROJECT"),
                google_location=os.getenv("GOOGLE_LOCATION"),
                google_credentials=os.getenv("GOOGLE_CREDENTIALS"),
            )
        elif provider == "ollama":
            return ClientFactory.create_ollama_client(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "llama2"),
                temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "4000")),
            )
        elif provider == "cohere":
            return ClientFactory.create_cohere_client(
                api_key=api_key,
                base_url=os.getenv("AI_BASE_URL"),
                model=os.getenv("AI_MODEL", "command-r-plus"),
                temperature=float(os.getenv("AI_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("AI_MAX_TOKENS", "4000")),
            )
        elif provider == "local":
            return ClientFactory.create_local_client(
                base_url=os.getenv("LOCAL_AI_BASE_URL", "http://localhost:8000"),
                model=os.getenv("AI_MODEL", "gpt-4"),
                temperature=float(os.getenv("AI_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("AI_MAX_TOKENS", "4000")),
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")


# Global client instance
_default_client: Optional[UnifiedClient] = None


def get_default_client() -> UnifiedClient:
    """Get default client instance"""
    global _default_client
    if _default_client is None:
        _default_client = ClientFactory.create_from_env()
    return _default_client


def set_default_client(client: UnifiedClient):
    """Set default client instance"""
    global _default_client
    _default_client = client
