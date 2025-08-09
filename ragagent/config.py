import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from .models import PipelineConfig, AgentConfig, DocumentProcessingConfig
from .utils import ConfigManager as EnhancedConfigManager, ValidationResult

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manager for configuration files and settings - Legacy wrapper"""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".ragagent"
        self.config_dir.mkdir(exist_ok=True)

        self.config_file = self.config_dir / "config.json"
        self.enhanced_manager = EnhancedConfigManager(self.config_file)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Default configuration
        self.default_config = {
            "agent": {
                "model_name": "qwen2.5:latest",
                "max_tokens": 4000,
                "temperature": 0.1,
                "timeout": 30,
                "base_url": "http://localhost:11434",
                "format": "json",  # qwen3 needs format parameter
            },
            "document_processing": {
                "max_chunk_size": 2000,
                "min_chunk_size": 100,
                "overlap_tokens": 100,
                "max_initial_chunks": 20,
                "max_navigation_depth": 3,
            },
            "enable_verification": True,
            "max_parallel_requests": 3,
            "cache": {
                "memory_size": 1000,
                "enable_disk_cache": False,
                "disk_dir": None,
                "disk_size_mb": 1000,
                "default_ttl": 3600,
            },
            "pool": {
                "max_connections": 10,
                "max_concurrent_requests": 5,
                "enable_priority": True,
            },
            "logging": {
                "level": "INFO",
                "file": "ragagent.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "version": "0.1.0",
        }

    def load_config(self) -> PipelineConfig:
        """Load configuration from file or create default"""
        try:
            return self.enhanced_manager.load_config()
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Using default configuration")
            return PipelineConfig(**self.default_config)

    def save_config(self, config: PipelineConfig):
        """Save configuration to file"""
        try:
            self.enhanced_manager.save_config(config)
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        try:
            return self.enhanced_manager.update_config(updates)
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            raise

    def reset_config(self):
        """Reset configuration to defaults"""
        try:
            default_config = PipelineConfig(**self.default_config)
            self.enhanced_manager.save_config(default_config)
            self.logger.info("Configuration reset to defaults")
            return default_config
        except Exception as e:
            self.logger.error(f"Error resetting configuration: {e}")
            raise

    def get_config_path(self) -> Path:
        """Get the path to the configuration file"""
        return self.enhanced_manager.config_file

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration dictionary"""
        try:
            validation_result = self.enhanced_manager.validator.validate_config(config)
            return validation_result.is_valid
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def export_config(self, output_path: Union[str, Path]):
        """Export configuration to a file"""
        try:
            original_config_file = self.enhanced_manager.config_file
            self.enhanced_manager.config_file = Path(output_path)
            config = self.enhanced_manager.load_config()
            self.enhanced_manager.save_config(config)
            self.enhanced_manager.config_file = original_config_file
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            raise

    def import_config(self, input_path: Union[str, Path]):
        """Import configuration from a file"""
        try:
            original_config_file = self.enhanced_manager.config_file
            self.enhanced_manager.config_file = Path(input_path)
            config = self.enhanced_manager.load_config()
            self.enhanced_manager.config_file = original_config_file
            self.enhanced_manager.save_config(config)
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            raise

    def get_env_config(self) -> Dict[str, Any]:
        """Get configuration from environment variables"""
        env_config = {}

        # Agent configuration
        if os.getenv("OLLAMA_MODEL"):
            env_config.setdefault("agent", {})["model_name"] = os.getenv("OLLAMA_MODEL")

        if os.getenv("OLLAMA_BASE_URL"):
            env_config.setdefault("agent", {})["base_url"] = os.getenv(
                "OLLAMA_BASE_URL"
            )

        if os.getenv("OLLAMA_TEMPERATURE"):
            env_config.setdefault("agent", {})["temperature"] = float(
                os.getenv("OLLAMA_TEMPERATURE")
            )

        if os.getenv("OLLAMA_MAX_TOKENS"):
            env_config.setdefault("agent", {})["max_tokens"] = int(
                os.getenv("OLLAMA_MAX_TOKENS")
            )

        if os.getenv("OLLAMA_TIMEOUT"):
            env_config.setdefault("agent", {})["timeout"] = int(
                os.getenv("OLLAMA_TIMEOUT")
            )

        # Document processing configuration
        if os.getenv("RAG_MAX_CHUNK_SIZE"):
            env_config.setdefault("document_processing", {})["max_chunk_size"] = int(
                os.getenv("RAG_MAX_CHUNK_SIZE")
            )

        if os.getenv("RAG_MAX_NAVIGATION_DEPTH"):
            env_config.setdefault("document_processing", {})["max_navigation_depth"] = (
                int(os.getenv("RAG_MAX_NAVIGATION_DEPTH"))
            )

        # Cache configuration
        if os.getenv("RAG_CACHE_MEMORY_SIZE"):
            env_config.setdefault("cache", {})["memory_size"] = int(
                os.getenv("RAG_CACHE_MEMORY_SIZE")
            )

        if os.getenv("RAG_CACHE_DISK_SIZE"):
            env_config.setdefault("cache", {})["disk_size_mb"] = int(
                os.getenv("RAG_CACHE_DISK_SIZE")
            )

        # Pool configuration
        if os.getenv("RAG_POOL_MAX_CONNECTIONS"):
            env_config.setdefault("pool", {})["max_connections"] = int(
                os.getenv("RAG_POOL_MAX_CONNECTIONS")
            )

        if os.getenv("RAG_POOL_MAX_CONCURRENT"):
            env_config.setdefault("pool", {})["max_concurrent_requests"] = int(
                os.getenv("RAG_POOL_MAX_CONCURRENT")
            )

        # Verification
        if os.getenv("RAG_ENABLE_VERIFICATION"):
            env_config["enable_verification"] = (
                os.getenv("RAG_ENABLE_VERIFICATION").lower() == "true"
            )

        return env_config

    def apply_env_config(self) -> PipelineConfig:
        """Apply environment variable configuration to current config"""
        try:
            env_config = self.get_env_config()

            if env_config:
                current_config = self.load_config()
                updated_config = self.update_config(env_config)
                self.logger.info("Applied environment configuration")
                return updated_config
            else:
                self.logger.info("No environment configuration found")
                return self.load_config()

        except Exception as e:
            self.logger.error(f"Error applying environment configuration: {e}")
            return self.load_config()

    def get_config_template(self) -> str:
        """Get a template configuration file"""
        return self.enhanced_manager.get_config_template()

    def validate_config_file(self) -> ValidationResult:
        """Validate the configuration file"""
        return self.enhanced_manager.validate_config_file()


class EnvironmentConfig:
    """Configuration management through environment variables"""

    @staticmethod
    def get_model_name() -> str:
        """Get model name from environment"""
        return os.getenv("OLLAMA_MODEL", "qwen3:latest")

    @staticmethod
    def get_base_url() -> str:
        """Get Ollama base URL from environment"""
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    @staticmethod
    def get_temperature() -> float:
        """Get temperature from environment"""
        return float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

    @staticmethod
    def get_max_tokens() -> int:
        """Get max tokens from environment"""
        return int(os.getenv("OLLAMA_MAX_TOKENS", "4000"))

    @staticmethod
    def get_timeout() -> int:
        """Get timeout from environment"""
        return int(os.getenv("OLLAMA_TIMEOUT", "30"))

    @staticmethod
    def get_max_chunk_size() -> int:
        """Get max chunk size from environment"""
        return int(os.getenv("RAG_MAX_CHUNK_SIZE", "2000"))

    @staticmethod
    def get_max_navigation_depth() -> int:
        """Get max navigation depth from environment"""
        return int(os.getenv("RAG_MAX_NAVIGATION_DEPTH", "3"))

    @staticmethod
    def get_enable_verification() -> bool:
        """Get enable verification from environment"""
        return os.getenv("RAG_ENABLE_VERIFICATION", "true").lower() == "true"

    @staticmethod
    def get_logging_level() -> str:
        """Get logging level from environment"""
        return os.getenv("RAG_LOG_LEVEL", "INFO")

    @staticmethod
    def get_cache_memory_size() -> int:
        """Get cache memory size from environment"""
        return int(os.getenv("RAG_CACHE_MEMORY_SIZE", "1000"))

    @staticmethod
    def get_cache_disk_size() -> int:
        """Get cache disk size from environment"""
        return int(os.getenv("RAG_CACHE_DISK_SIZE", "1000"))

    @staticmethod
    def get_pool_max_connections() -> int:
        """Get pool max connections from environment"""
        return int(os.getenv("RAG_POOL_MAX_CONNECTIONS", "10"))

    @staticmethod
    def get_pool_max_concurrent() -> int:
        """Get pool max concurrent requests from environment"""
        return int(os.getenv("RAG_POOL_MAX_CONCURRENT", "5"))


def create_config_from_env() -> PipelineConfig:
    """Create PipelineConfig from environment variables"""
    return PipelineConfig(
        agent=AgentConfig(
            model_name=EnvironmentConfig.get_model_name(),
            base_url=EnvironmentConfig.get_base_url(),
            temperature=EnvironmentConfig.get_temperature(),
            max_tokens=EnvironmentConfig.get_max_tokens(),
            timeout=EnvironmentConfig.get_timeout(),
        ),
        document_processing=DocumentProcessingConfig(
            max_chunk_size=EnvironmentConfig.get_max_chunk_size(),
            max_navigation_depth=EnvironmentConfig.get_max_navigation_depth(),
        ),
        enable_verification=EnvironmentConfig.get_enable_verification(),
        cache={
            "memory_size": EnvironmentConfig.get_cache_memory_size(),
            "disk_size_mb": EnvironmentConfig.get_cache_disk_size(),
        },
        pool={
            "max_connections": EnvironmentConfig.get_pool_max_connections(),
            "max_concurrent_requests": EnvironmentConfig.get_pool_max_concurrent(),
        },
    )


def setup_logging_from_config(config: Dict[str, Any]):
    """Setup logging from configuration"""
    logging_config = config.get("logging", {})

    log_level = logging_config.get("level", "INFO")
    log_file = logging_config.get("file", "ragagent.log")
    log_format = logging_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logger.info(f"Logging configured - Level: {log_level}, File: {log_file}")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> PipelineConfig:
    """Get the current configuration"""
    return config_manager.load_config()


def update_config(updates: Dict[str, Any]) -> PipelineConfig:
    """Update configuration"""
    return config_manager.update_config(updates)


def reset_config() -> PipelineConfig:
    """Reset configuration to defaults"""
    return config_manager.reset_config()


def validate_config_file() -> ValidationResult:
    """Validate the configuration file"""
    return config_manager.validate_config_file()


def get_config_template() -> str:
    """Get a template configuration file"""
    return config_manager.get_config_template()
