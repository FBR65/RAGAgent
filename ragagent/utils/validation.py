"""
Enhanced validation and configuration management for the Agentic RAG system
"""

import logging
import re
import json
from typing import Any, Dict, List, Optional, Union, Type, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import pydantic
from pydantic import BaseModel, validator, Field, ValidationError
import yaml

from ..models import PipelineConfig, AgentConfig, DocumentProcessingConfig

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation operation"""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    infos: List[str] = field(default_factory=list)
    validated_data: Optional[Dict[str, Any]] = None

    def add_error(self, message: str):
        """Add an error message"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """Add a warning message"""
        self.warnings.append(message)

    def add_info(self, message: str):
        """Add an info message"""
        self.infos.append(message)

    def get_messages(self, severity: ValidationSeverity = None) -> List[str]:
        """Get messages by severity"""
        if severity == ValidationSeverity.ERROR:
            return self.errors
        elif severity == ValidationSeverity.WARNING:
            return self.warnings
        elif severity == ValidationSeverity.INFO:
            return self.infos
        else:
            return self.errors + self.warnings + self.infos


class ValidationRule:
    """Base class for validation rules"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate a value against the rule"""
        raise NotImplementedError


class RequiredRule(ValidationRule):
    """Rule to check if a value is required"""

    def __init__(self, name: str = "required"):
        super().__init__(name, "Value is required")

    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        result = ValidationResult(True)

        if value is None or value == "":
            result.add_error("Value is required")

        return result


class TypeRule(ValidationRule):
    """Rule to check if a value is of a specific type"""

    def __init__(self, expected_type: Type, name: str = "type_check"):
        super().__init__(name, f"Value must be of type {expected_type.__name__}")
        self.expected_type = expected_type

    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        result = ValidationResult(True)

        if not isinstance(value, self.expected_type):
            result.add_error(
                f"Expected {self.expected_type.__name__}, got {type(value).__name__}"
            )

        return result


class RangeRule(ValidationRule):
    """Rule to check if a numeric value is within a range"""

    def __init__(
        self,
        min_value: float = None,
        max_value: float = None,
        name: str = "range_check",
    ):
        super().__init__(name, f"Value must be between {min_value} and {max_value}")
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        result = ValidationResult(True)

        if not isinstance(value, (int, float)):
            result.add_error("Value must be numeric")
            return result

        if self.min_value is not None and value < self.min_value:
            result.add_error(f"Value {value} is less than minimum {self.min_value}")

        if self.max_value is not None and value > self.max_value:
            result.add_error(f"Value {value} is greater than maximum {self.max_value}")

        return result


class RegexRule(ValidationRule):
    """Rule to check if a string matches a regex pattern"""

    def __init__(self, pattern: str, name: str = "regex_check"):
        super().__init__(name, f"Value must match pattern: {pattern}")
        self.pattern = re.compile(pattern)

    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        result = ValidationResult(True)

        if not isinstance(value, str):
            result.add_error("Value must be a string")
            return result

        if not self.pattern.match(value):
            result.add_error(f"Value '{value}' does not match pattern")

        return result


class EnumRule(ValidationRule):
    """Rule to check if a value is in a specific set of allowed values"""

    def __init__(self, allowed_values: List[Any], name: str = "enum_check"):
        super().__init__(name, f"Value must be one of: {allowed_values}")
        self.allowed_values = allowed_values

    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        result = ValidationResult(True)

        if value not in self.allowed_values:
            result.add_error(
                f"Value '{value}' is not allowed. Must be one of: {self.allowed_values}"
            )

        return result


class CustomRule(ValidationRule):
    """Rule with custom validation function"""

    def __init__(
        self,
        validator_func: Callable[[Any], bool],
        error_message: str,
        name: str = "custom_check",
    ):
        super().__init__(name, error_message)
        self.validator_func = validator_func

    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        result = ValidationResult(True)

        if not self.validator_func(value):
            result.add_error(self.description)

        return result


class SchemaValidator:
    """Validates configuration against a schema"""

    def __init__(self):
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_rule(self, field_path: str, rule: ValidationRule):
        """Add a validation rule for a field path"""
        if field_path not in self.rules:
            self.rules[field_path] = []
        self.rules[field_path].append(rule)

    def validate(
        self, data: Dict[str, Any], context: Dict[str, Any] = None
    ) -> ValidationResult:
        """Validate data against all rules"""
        result = ValidationResult(True)
        context = context or {}

        for field_path, rules in self.rules.items():
            # Get the value using dot notation
            value = self._get_nested_value(data, field_path)

            # Validate against each rule
            for rule in rules:
                rule_result = rule.validate(value, context)
                result.errors.extend(rule_result.errors)
                result.warnings.extend(rule_result.warnings)
                result.infos.extend(rule_result.infos)

        if result.errors:
            result.is_valid = False

        return result

    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get a nested value using dot notation"""
        keys = field_path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current


class ConfigValidator:
    """Main configuration validator"""

    def __init__(self):
        self.schema_validator = SchemaValidator()
        self._setup_rules()

    def _setup_rules(self):
        """Setup validation rules for configuration"""

        # Agent configuration rules
        self.schema_validator.add_rule("agent.model_name", RequiredRule())
        self.schema_validator.add_rule(
            "agent.model_name", RegexRule(r"^[a-zA-Z0-9_\-:.]+$", "model_name")
        )
        self.schema_validator.add_rule("agent.base_url", RequiredRule())
        self.schema_validator.add_rule(
            "agent.base_url", RegexRule(r"^https?://.+", "base_url")
        )
        self.schema_validator.add_rule("agent.temperature", RangeRule(0.0, 2.0))
        self.schema_validator.add_rule("agent.max_tokens", RangeRule(1, 32000))
        self.schema_validator.add_rule("agent.timeout", RangeRule(1, 300))

        # Document processing rules
        self.schema_validator.add_rule(
            "document_processing.max_chunk_size", RangeRule(100, 10000)
        )
        self.schema_validator.add_rule(
            "document_processing.min_chunk_size", RangeRule(10, 5000)
        )
        self.schema_validator.add_rule(
            "document_processing.max_navigation_depth", RangeRule(1, 10)
        )
        self.schema_validator.add_rule(
            "document_processing.max_initial_chunks", RangeRule(1, 100)
        )

        # Cache rules
        self.schema_validator.add_rule("cache.memory_size", RangeRule(100, 100000))
        self.schema_validator.add_rule("cache.disk_size_mb", RangeRule(10, 10000))
        self.schema_validator.add_rule("cache.default_ttl", RangeRule(60, 86400))

        # Pool rules
        self.schema_validator.add_rule("pool.max_connections", RangeRule(1, 100))
        self.schema_validator.add_rule("pool.max_concurrent_requests", RangeRule(1, 50))

        # Enable flags should be boolean
        self.schema_validator.add_rule(
            "enable_verification",
            CustomRule(
                lambda x: isinstance(x, bool), "enable_verification must be boolean"
            ),
        )
        self.schema_validator.add_rule(
            "cache.enable_disk_cache",
            CustomRule(
                lambda x: isinstance(x, bool), "enable_disk_cache must be boolean"
            ),
        )
        self.schema_validator.add_rule(
            "pool.enable_priority",
            CustomRule(
                lambda x: isinstance(x, bool), "enable_priority must be boolean"
            ),
        )

    def validate_config(
        self, config: Union[Dict[str, Any], PipelineConfig]
    ) -> ValidationResult:
        """Validate a configuration"""

        # Convert to dict if it's a PipelineConfig
        if isinstance(config, PipelineConfig):
            config_dict = config.dict()
        else:
            config_dict = config

        result = self.schema_validator.validate(config_dict)

        # Additional validation logic
        self._validate_agent_config(config_dict, result)
        self._validate_cache_config(config_dict, result)
        self._validate_pool_config(config_dict, result)

        return result

    def _validate_agent_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate agent-specific configuration"""

        # Check if model name is compatible with base URL
        if "agent" in config:
            agent_config = config["agent"]
            model_name = agent_config.get("model_name", "")
            base_url = agent_config.get("base_url", "")

            # Ollama-specific validation
            if "ollama" in base_url.lower():
                if not model_name:
                    result.add_error("model_name is required for Ollama")
                elif ":" not in model_name:
                    result.add_warning(
                        "Ollama model names should include version (e.g., 'llama2:latest')"
                    )

            # OpenAI-specific validation
            elif "openai" in base_url.lower():
                if not model_name.startswith(("gpt-", "text-")):
                    result.add_warning(
                        f"OpenAI model names typically start with 'gpt-' or 'text-', got: {model_name}"
                    )

    def _validate_cache_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate cache-specific configuration"""

        if "cache" in config:
            cache_config = config["cache"]

            # Check if disk cache is enabled but no directory specified
            enable_disk = cache_config.get("enable_disk_cache", False)
            disk_dir = cache_config.get("disk_dir", None)

            if enable_disk and not disk_dir:
                result.add_warning("Disk cache is enabled but no directory specified")

            # Check memory size vs disk size
            memory_size = cache_config.get("memory_size", 0)
            disk_size = cache_config.get("disk_size_mb", 0)

            if memory_size > disk_size * 10:  # Arbitrary ratio
                result.add_warning(
                    "Memory cache size is much larger than disk cache size"
                )

    def _validate_pool_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate pool-specific configuration"""

        if "pool" in config:
            pool_config = config["pool"]

            # Check if max connections is less than max concurrent requests
            max_connections = pool_config.get("max_connections", 0)
            max_concurrent = pool_config.get("max_concurrent_requests", 0)

            if max_concurrent > max_connections:
                result.add_warning(
                    "Max concurrent requests should not exceed max connections"
                )


class ConfigMigrator:
    """Handles configuration migration between versions"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.migrations = {
            "0.1.0": self._migrate_from_0_1_0,
        }

    def migrate_config(
        self, config: Dict[str, Any], from_version: str, to_version: str
    ) -> Dict[str, Any]:
        """Migrate configuration from one version to another"""

        current_version = from_version

        while current_version != to_version:
            if current_version in self.migrations:
                config = self.migrations[current_version](config)
                current_version = "0.1.0"  # All migrations go to 0.1.0
            else:
                self.logger.warning(f"No migration found for version {current_version}")
                break

        return config

    def _migrate_from_0_1_0(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 0.1.0 (current version)"""
        # No migrations needed for now
        return config

    def get_config_version(self, config: Dict[str, Any]) -> str:
        """Get the version of a configuration"""
        return config.get("version", "0.1.0")


class ConfigManager:
    """Enhanced configuration manager with validation"""

    def __init__(self, config_file: Path = None):
        self.config_file = config_file or Path("ragagent_config.yaml")
        self.validator = ConfigValidator()
        self.migrator = ConfigMigrator()
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_config(self) -> PipelineConfig:
        """Load configuration with validation"""

        try:
            if not self.config_file.exists():
                self.logger.info("No configuration file found, using defaults")
                return self._create_default_config()

            with open(self.config_file, "r", encoding="utf-8") as f:
                if self.config_file.suffix.lower() == ".yaml":
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            # Validate configuration
            validation_result = self.validator.validate_config(config_data)

            if not validation_result.is_valid:
                self.logger.error(
                    f"Configuration validation failed: {validation_result.errors}"
                )
                raise ValidationError(
                    f"Invalid configuration: {validation_result.errors}"
                )

            # Log warnings
            for warning in validation_result.warnings:
                self.logger.warning(f"Configuration warning: {warning}")

            # Create PipelineConfig
            pipeline_config = PipelineConfig(**config_data)

            self.logger.info(
                f"Configuration loaded successfully from {self.config_file}"
            )
            return pipeline_config

        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Using default configuration")
            return self._create_default_config()

    def save_config(self, config: PipelineConfig):
        """Save configuration with validation"""

        try:
            # Convert to dict
            config_dict = config.dict()

            # Validate before saving
            validation_result = self.validator.validate_config(config_dict)

            if not validation_result.is_valid:
                self.logger.error(
                    f"Cannot save invalid configuration: {validation_result.errors}"
                )
                raise ValidationError(
                    f"Invalid configuration: {validation_result.errors}"
                )

            # Create directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            # Save configuration
            with open(self.config_file, "w", encoding="utf-8") as f:
                if self.config_file.suffix.lower() == ".yaml":
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Configuration saved to {self.config_file}")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise

    def _create_default_config(self) -> PipelineConfig:
        """Create default configuration"""
        return PipelineConfig(
            agent=AgentConfig(
                model_name="qwen3:latest",
                base_url="http://localhost:11434",
                temperature=0.1,
                max_tokens=4000,
                timeout=30,
            ),
            document_processing=DocumentProcessingConfig(
                max_chunk_size=2000,
                min_chunk_size=100,
                overlap_tokens=100,
                max_initial_chunks=20,
                max_navigation_depth=3,
            ),
            enable_verification=True,
            cache={
                "memory_size": 1000,
                "enable_disk_cache": False,
                "disk_dir": None,
                "disk_size_mb": 1000,
                "default_ttl": 3600,
            },
            pool={
                "max_connections": 10,
                "max_concurrent_requests": 5,
                "enable_priority": True,
            },
            version="0.1.0",
        )

    def validate_config_file(self) -> ValidationResult:
        """Validate the configuration file"""

        if not self.config_file.exists():
            return ValidationResult(
                False, [f"Configuration file not found: {self.config_file}"]
            )

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                if self.config_file.suffix.lower() == ".yaml":
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            return self.validator.validate_config(config_data)

        except Exception as e:
            return ValidationResult(False, [f"Error reading configuration file: {e}"])

    def get_config_template(self) -> str:
        """Get a template configuration file"""
        template = {
            "version": "0.1.0",
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
        }

        return json.dumps(template, indent=2, ensure_ascii=False)


# Global configuration manager instance
global_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager"""
    return global_config_manager


def validate_config(config: Union[Dict[str, Any], PipelineConfig]) -> ValidationResult:
    """Validate a configuration using the global validator"""
    return global_config_manager.validator.validate_config(config)
