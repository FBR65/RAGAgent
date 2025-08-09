import httpx
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from .models import AgentConfig

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI-compatible client for interacting with OpenAI API or compatible services"""

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
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {config.api_key}"}
                    if hasattr(config, "api_key") and config.api_key
                    else {}
                ),
            },
        )

        logger.info(f"Initialized OpenAI client with model: {self.model_name}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Make a chat completion request to OpenAI-compatible API"""

        try:
            # Use standard OpenAI API for all models
            if "ollama" not in self.base_url.lower():
                # For qwen2.5, we need to use the native Ollama API format
                # Extract the user message
                user_message = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_message = msg.get("content", "")
                        break

                if not user_message:
                    user_message = messages[-1].get("content", "") if messages else ""

                # Use OpenAI-compatible API format
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": user_message}],
                    "temperature": temperature
                    if temperature is not None
                    else self.temperature,
                    "stream": False,
                }

                if max_tokens:
                    payload["max_tokens"] = max_tokens

                logger.debug(
                    f"Sending request to Ollama: {json.dumps(payload, indent=2)}"
                )

                # Use standard OpenAI endpoint
                response = self.client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()

                result = response.json()
                logger.debug(
                    f"Received response from OpenAI: {json.dumps(result, indent=2)}"
                )

                return result
            else:
                # Standard OpenAI format for other models
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
                    # OpenAI expects response_format as dict, not string
                    payload["response_format"] = response_format

                if max_tokens:
                    payload["max_tokens"] = max_tokens

                logger.debug(
                    f"Sending request to OpenAI: {json.dumps(payload, indent=2)}"
                )

                response = self.client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()

                result = response.json()
                logger.debug(
                    f"Received response from OpenAI: {json.dumps(result, indent=2)}"
                )

                return result

        except httpx.HTTPError as e:
            logger.error(f"HTTP error from OpenAI: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error from OpenAI: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error from OpenAI: {e}")
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

            logger.debug(f"Sending generate request to OpenAI")

            response = self.client.post("/v1/completions", json=payload)
            response.raise_for_status()

            result = response.json()
            logger.debug(f"Received generate response from OpenAI")

            return result.get("choices", [{}])[0].get("text", "")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error from OpenAI: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error from OpenAI: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error from OpenAI: {e}")
            raise

    def check_model_availability(self) -> bool:
        """Check if the model is available"""
        try:
            response = self.client.get("/v1/models")
            response.raise_for_status()

            models = response.json().get("data", [])
            available_models = [model.get("id") for model in models]

            is_available = self.model_name in available_models
            logger.info(f"Model {self.model_name} available: {is_available}")

            return is_available

        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the model"""
        try:
            response = self.client.get("/v1/models")
            response.raise_for_status()

            models = response.json().get("data", [])
            for model in models:
                if model.get("id") == self.model_name:
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


class OpenAIClientFactory:
    """Factory for creating OpenAI clients with configuration"""

    @staticmethod
    def create_client(config: Optional[AgentConfig] = None) -> OpenAIClient:
        """Create an OpenAI client with the given configuration"""
        if config is None:
            config = AgentConfig()

        return OpenAIClient(config)

    @staticmethod
    def create_default_client() -> OpenAIClient:
        """Create an OpenAI client with default configuration"""
        config = AgentConfig()
        return OpenAIClient(config)

    @staticmethod
    def create_custom_client(
        model_name: str = "gpt-3.5-turbo",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = None,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        timeout: int = 30,
    ) -> OpenAIClient:
        """Create an OpenAI client with custom parameters"""
        config = AgentConfig(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        if api_key:
            config.api_key = api_key

        return OpenAIClient(config)
