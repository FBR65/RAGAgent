import httpx
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from .models import AgentConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.base_url = config.base_url
        self.model_name = config.model_name
        self.timeout = config.timeout
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        # Initialize HTTP client
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            headers={"Content-Type": "application/json"},
        )

        logger.info(f"Initialized Ollama client with model: {self.model_name}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Make a chat completion request to Ollama"""

        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature
                if temperature is not None
                else self.temperature,
                "stream": False,
            }

            if tools:
                payload["tools"] = tools

            if tool_choice:
                payload["tool_choice"] = tool_choice

            if response_format:
                # Ollama expects format as string, not dict
                if isinstance(response_format, dict):
                    payload["format"] = json.dumps(response_format)
                else:
                    payload["format"] = response_format

            if max_tokens:
                payload["max_tokens"] = max_tokens

            logger.debug(f"Sending request to Ollama: {json.dumps(payload, indent=2)}")

            response = self.client.post("/api/chat", json=payload)
            response.raise_for_status()

            result = response.json()
            logger.debug(
                f"Received response from Ollama: {json.dumps(result, indent=2)}"
            )

            return result

        except httpx.HTTPError as e:
            logger.error(f"HTTP error from Ollama: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error from Ollama: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error from Ollama: {e}")
            raise

    def generate_completion(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a simple completion without chat history"""

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature
                if temperature is not None
                else self.temperature,
                "stream": False,
            }

            if max_tokens:
                payload["max_tokens"] = max_tokens

            logger.debug(f"Sending generate request to Ollama")

            response = self.client.post("/api/generate", json=payload)
            response.raise_for_status()

            result = response.json()
            logger.debug(f"Received generate response from Ollama")

            return result.get("response", "")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error from Ollama: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error from Ollama: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error from Ollama: {e}")
            raise

    def check_model_availability(self) -> bool:
        """Check if the model is available"""
        try:
            response = self.client.get("/api/tags")
            response.raise_for_status()

            models = response.json().get("models", [])
            available_models = [model.get("name") for model in models]

            is_available = self.model_name in available_models
            logger.info(f"Model {self.model_name} available: {is_available}")

            return is_available

        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the model"""
        try:
            response = self.client.get("/api/tags")
            response.raise_for_status()

            models = response.json().get("models", [])
            for model in models:
                if model.get("name") == self.model_name:
                    return model

            return None

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None

    def close(self):
        """Close the HTTP client"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class OllamaClientFactory:
    """Factory for creating Ollama clients with configuration"""

    @staticmethod
    def create_client(config: Optional[AgentConfig] = None) -> OllamaClient:
        """Create an Ollama client with the given configuration"""
        if config is None:
            config = AgentConfig()

        return OllamaClient(config)

    @staticmethod
    def create_default_client() -> OllamaClient:
        """Create an Ollama client with default configuration"""
        config = AgentConfig()
        return OllamaClient(config)

    @staticmethod
    def create_custom_client(
        model_name: str = "qwen3:latest",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        timeout: int = 30,
    ) -> OllamaClient:
        """Create an Ollama client with custom parameters"""
        config = AgentConfig(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return OllamaClient(config)
