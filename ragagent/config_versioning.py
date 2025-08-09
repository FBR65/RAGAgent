"""
Configuration versioning system for the RAG agent
"""

import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import copy
from enum import Enum

from .models import PipelineConfig, AgentConfig, DocumentProcessingConfig

logger = logging.getLogger(__name__)


class ConfigVersion(Enum):
    """Configuration versions"""

    V1 = "v1.0.0"
    V2 = "v2.0.0"
    V3 = "v3.0.0"
    V4 = "v4.0.0"
    V5 = "v5.0.0"
    CURRENT = "current"


@dataclass
class ConfigMetadata:
    """Configuration metadata"""

    version: str
    created_at: str
    created_by: str
    description: str
    is_active: bool = False
    is_current: bool = False
    hash: Optional[str] = None
    parent_version: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

        # Generate hash if not provided
        if self.hash is None:
            self.hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate hash for configuration"""
        config_dict = asdict(self)
        config_dict.pop("hash", None)  # Remove hash field from calculation
        config_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigMetadata":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ConfigVersionInfo:
    """Configuration version information"""

    version: str
    metadata: ConfigMetadata
    config_data: Dict[str, Any]
    migration_log: List[str] = None

    def __post_init__(self):
        if self.migration_log is None:
            self.migration_log = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "metadata": self.metadata.to_dict(),
            "config_data": self.config_data,
            "migration_log": self.migration_log,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigVersionInfo":
        """Create from dictionary"""
        return cls(
            version=data["version"],
            metadata=ConfigMetadata.from_dict(data["metadata"]),
            config_data=data["config_data"],
            migration_log=data.get("migration_log", []),
        )


class ConfigMigrator:
    """Configuration migrator"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def migrate_config(
        self, config_data: Dict[str, Any], from_version: str, to_version: str
    ) -> Dict[str, Any]:
        """Migrate configuration from one version to another"""
        self.logger.info(f"Migrating config from {from_version} to {to_version}")

        # Start with original config
        migrated_config = copy.deepcopy(config_data)
        migration_log = []

        # Apply migrations in order
        if (
            from_version == ConfigVersion.V1.value
            and to_version >= ConfigVersion.V2.value
        ):
            migrated_config, log = self._migrate_v1_to_v2(migrated_config)
            migration_log.extend(log)

        if (
            from_version == ConfigVersion.V2.value
            and to_version >= ConfigVersion.V3.value
        ):
            migrated_config, log = self._migrate_v2_to_v3(migrated_config)
            migration_log.extend(log)

        if (
            from_version == ConfigVersion.V3.value
            and to_version >= ConfigVersion.V4.value
        ):
            migrated_config, log = self._migrate_v3_to_v4(migrated_config)
            migration_log.extend(log)

        if (
            from_version == ConfigVersion.V4.value
            and to_version >= ConfigVersion.V5.value
        ):
            migrated_config, log = self._migrate_v4_to_v5(migrated_config)
            migration_log.extend(log)

        self.logger.info(f"Migration completed with {len(migration_log)} changes")
        return migrated_config, migration_log

    def _migrate_v1_to_v2(
        self, config_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[str]]:
        """Migrate from v1.0.0 to v2.0.0"""
        migration_log = []

        # Add new agent configuration structure
        if "agent" in config_data and isinstance(config_data["agent"], dict):
            old_agent = config_data["agent"]

            # Create new agent config structure
            new_agent = {
                "model_name": old_agent.get("model", "gpt-4"),
                "base_url": old_agent.get("base_url", "https://api.openai.com/v1"),
                "temperature": old_agent.get("temperature", 0.1),
                "max_tokens": old_agent.get("max_tokens", 4000),
                "timeout": old_agent.get("timeout", 30),
                "format": old_agent.get("format", "json"),
            }

            config_data["agent"] = new_agent
            migration_log.append("Restructured agent configuration")

        # Add document processing configuration
        if "document_processing" not in config_data:
            config_data["document_processing"] = {
                "max_chunk_size": 2000,
                "min_chunk_size": 100,
                "overlap_tokens": 100,
                "max_initial_chunks": 20,
                "max_navigation_depth": 3,
            }
            migration_log.append("Added document processing configuration")

        # Add enable_verification flag
        if "enable_verification" not in config_data:
            config_data["enable_verification"] = True
            migration_log.append("Added enable_verification flag")

        return config_data, migration_log

    def _migrate_v2_to_v3(
        self, config_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[str]]:
        """Migrate from v2.0.0 to v3.0.0"""
        migration_log = []

        # Add logging configuration
        if "logging" not in config_data:
            config_data["logging"] = {
                "level": "INFO",
                "file": "ragagent.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            }
            migration_log.append("Added logging configuration")

        # Add max_parallel_requests
        if "max_parallel_requests" not in config_data:
            config_data["max_parallel_requests"] = 3
            migration_log.append("Added max_parallel_requests")

        # Update agent configuration with new fields
        if "agent" in config_data and isinstance(config_data["agent"], dict):
            agent = config_data["agent"]

            # Add new fields
            if "retry_count" not in agent:
                agent["retry_count"] = 3
                migration_log.append("Added retry_count to agent config")

            if "retry_delay" not in agent:
                agent["retry_delay"] = 1.0
                migration_log.append("Added retry_delay to agent config")

        return config_data, migration_log

    def _migrate_v3_to_v4(
        self, config_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[str]]:
        """Migrate from v3.0.0 to v4.0.0"""
        migration_log = []

        # Add OCR configuration
        if "ocr" not in config_data:
            config_data["ocr"] = {
                "enabled": True,
                "mode": "paragraph_based",
                "quality": "high",
                "language": "eng",
                "dpi": 300,
                "enable_preprocessing": True,
                "enable_postprocessing": True,
                "enable_layout_analysis": True,
                "max_pages": 100,
                "timeout": 300,
                "retry_count": 3,
                "cache_enabled": True,
            }
            migration_log.append("Added OCR configuration")

        # Add resilience configuration
        if "resilience" not in config_data:
            config_data["resilience"] = {
                "timeout": {
                    "connect_timeout": 10.0,
                    "read_timeout": 30.0,
                    "write_timeout": 30.0,
                    "total_timeout": 60.0,
                },
                "retry": {
                    "max_retries": 3,
                    "base_delay": 1.0,
                    "max_delay": 60.0,
                    "strategy": "exponential_jitter",
                },
                "circuit_breaker": {
                    "failure_threshold": 5,
                    "recovery_timeout": 60,
                },
                "rate_limiting": {
                    "requests_per_second": 10,
                    "requests_per_minute": 600,
                    "requests_per_hour": 36000,
                    "burst_size": 5,
                },
            }
            migration_log.append("Added resilience configuration")

        return config_data, migration_log

    def _migrate_v4_to_v5(
        self, config_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[str]]:
        """Migrate from v4.0.0 to v5.0.0"""
        migration_log = []

        # Add hybrid RAG configuration
        if "hybrid_rag" not in config_data:
            config_data["hybrid_rag"] = {
                "enabled": False,
                "search_strategy": "hybrid",
                "vector_db": {
                    "provider": "qdrant",
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "rag_embeddings",
                },
                "embedder": {
                    "model": "BGE-large-en-v1.5",
                    "batch_size": 32,
                    "device": "cpu",
                },
                "fusion_weights": {
                    "vector": 0.5,
                    "keyword": 0.3,
                    "semantic": 0.2,
                },
            }
            migration_log.append("Added hybrid RAG configuration")

        # Add specialized agents configuration
        if "specialized_agents" not in config_data:
            config_data["specialized_agents"] = {
                "enabled": False,
                "default_domain": "general",
                "agents": {
                    "legal": {
                        "enabled": True,
                        "model": "gpt-4",
                        "temperature": 0.1,
                        "max_tokens": 4000,
                    },
                    "medical": {
                        "enabled": True,
                        "model": "gpt-4",
                        "temperature": 0.1,
                        "max_tokens": 4000,
                    },
                    "technical": {
                        "enabled": True,
                        "model": "gpt-4",
                        "temperature": 0.1,
                        "max_tokens": 4000,
                    },
                },
            }
            migration_log.append("Added specialized agents configuration")

        # Add learning system configuration
        if "learning_system" not in config_data:
            config_data["learning_system"] = {
                "enabled": False,
                "adaptation_threshold": 0.8,
                "learning_rate": 0.01,
                "max_history_size": 1000,
                "preference_learning": {
                    "enabled": True,
                    "update_frequency": "daily",
                },
            }
            migration_log.append("Added learning system configuration")

        return config_data, migration_log


class ConfigVersionManager:
    """Configuration version manager"""

    def __init__(self, config_dir: Union[str, Path] = "config_versions"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize migrator
        self.migrator = ConfigMigrator()

        # Version info storage
        self.versions_file = self.config_dir / "versions.json"
        self.versions: Dict[str, ConfigVersionInfo] = {}

        # Load existing versions
        self._load_versions()

    def _load_versions(self):
        """Load existing configuration versions"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, "r", encoding="utf-8") as f:
                    versions_data = json.load(f)

                for version_data in versions_data:
                    version_info = ConfigVersionInfo.from_dict(version_data)
                    self.versions[version_info.version] = version_info

                self.logger.info(f"Loaded {len(self.versions)} configuration versions")
            except Exception as e:
                self.logger.error(f"Error loading configuration versions: {e}")
                self.versions = {}

    def _save_versions(self):
        """Save configuration versions"""
        try:
            versions_data = [version.to_dict() for version in self.versions.values()]
            with open(self.versions_file, "w", encoding="utf-8") as f:
                json.dump(versions_data, f, indent=2, ensure_ascii=False)

            self.logger.info("Configuration versions saved")
        except Exception as e:
            self.logger.error(f"Error saving configuration versions: {e}")
            raise

    def save_version(
        self, config: PipelineConfig, metadata: ConfigMetadata, force: bool = False
    ) -> ConfigVersionInfo:
        """Save a new configuration version"""
        # Convert config to dict
        config_dict = config.dict()

        # Check if version already exists
        if metadata.version in self.versions and not force:
            raise ValueError(f"Version {metadata.version} already exists")

        # Create version info
        version_info = ConfigVersionInfo(
            version=metadata.version,
            metadata=metadata,
            config_data=config_dict,
        )

        # Save version
        self.versions[metadata.version] = version_info
        self._save_versions()

        self.logger.info(f"Configuration version {metadata.version} saved")
        return version_info

    def get_version(self, version: str) -> Optional[ConfigVersionInfo]:
        """Get configuration version"""
        return self.versions.get(version)

    def get_current_version(self) -> Optional[ConfigVersionInfo]:
        """Get current configuration version"""
        for version_info in self.versions.values():
            if version_info.metadata.is_current:
                return version_info
        return None

    def get_active_versions(self) -> List[ConfigVersionInfo]:
        """Get all active configuration versions"""
        return [v for v in self.versions.values() if v.metadata.is_active]

    def set_current_version(self, version: str) -> bool:
        """Set current configuration version"""
        if version not in self.versions:
            return False

        # Clear current flag from all versions
        for v in self.versions.values():
            v.metadata.is_current = False

        # Set current version
        self.versions[version].metadata.is_current = True
        self._save_versions()

        self.logger.info(f"Set current version to {version}")
        return True

    def activate_version(self, version: str) -> bool:
        """Activate configuration version"""
        if version not in self.versions:
            return False

        self.versions[version].metadata.is_active = True
        self._save_versions()

        self.logger.info(f"Activated version {version}")
        return True

    def deactivate_version(self, version: str) -> bool:
        """Deactivate configuration version"""
        if version not in self.versions:
            return False

        self.versions[version].metadata.is_active = False
        self._save_versions()

        self.logger.info(f"Deactivated version {version}")
        return True

    def delete_version(self, version: str) -> bool:
        """Delete configuration version"""
        if version not in self.versions:
            return False

        # Don't delete current version
        if self.versions[version].metadata.is_current:
            return False

        del self.versions[version]
        self._save_versions()

        self.logger.info(f"Deleted version {version}")
        return True

    def migrate_to_version(
        self, from_version: str, to_version: str
    ) -> tuple[Dict[str, Any], List[str]]:
        """Migrate configuration from one version to another"""
        if from_version not in self.versions:
            raise ValueError(f"Source version {from_version} not found")

        if to_version not in self.versions:
            raise ValueError(f"Target version {to_version} not found")

        # Get source configuration
        source_config = self.versions[from_version].config_data

        # Migrate configuration
        migrated_config, migration_log = self.migrator.migrate_config(
            source_config, from_version, to_version
        )

        return migrated_config, migration_log

    def create_from_current(
        self, description: str, created_by: str = "system"
    ) -> ConfigVersionInfo:
        """Create new version from current configuration"""
        # Get current configuration (this would need to be passed in or loaded)
        # For now, we'll create a default configuration
        current_config = PipelineConfig()

        # Create metadata
        metadata = ConfigMetadata(
            version=ConfigVersion.CURRENT.value,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            description=description,
            is_current=True,
            is_active=True,
        )

        return self.save_version(current_config, metadata)

    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get version history"""
        history = []
        for version_info in sorted(
            self.versions.values(), key=lambda x: x.metadata.created_at
        ):
            history.append(
                {
                    "version": version_info.version,
                    "created_at": version_info.metadata.created_at,
                    "created_by": version_info.metadata.created_by,
                    "description": version_info.metadata.description,
                    "is_active": version_info.metadata.is_active,
                    "is_current": version_info.metadata.is_current,
                    "hash": version_info.metadata.hash,
                }
            )

        return history

    def export_version(self, version: str, export_path: Union[str, Path]) -> bool:
        """Export configuration version to file"""
        if version not in self.versions:
            return False

        version_info = self.versions[version]
        export_path = Path(export_path)

        try:
            export_data = version_info.to_dict()
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Exported version {version} to {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting version {version}: {e}")
            return False

    def import_version(self, import_path: Union[str, Path]) -> bool:
        """Import configuration version from file"""
        import_path = Path(import_path)

        if not import_path.exists():
            return False

        try:
            with open(import_path, "r", encoding="utf-8") as f:
                import_data = json.load(f)

            version_info = ConfigVersionInfo.from_dict(import_data)

            # Check if version already exists
            if version_info.version in self.versions:
                self.logger.warning(f"Version {version_info.version} already exists")
                return False

            # Save version
            self.versions[version_info.version] = version_info
            self._save_versions()

            self.logger.info(
                f"Imported version {version_info.version} from {import_path}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error importing version from {import_path}: {e}")
            return False

    def get_diff(self, version1: str, version2: str) -> Dict[str, Any]:
        """Get differences between two configuration versions"""
        if version1 not in self.versions:
            raise ValueError(f"Version {version1} not found")

        if version2 not in self.versions:
            raise ValueError(f"Version {version2} not found")

        config1 = self.versions[version1].config_data
        config2 = self.versions[version2].config_data

        # Simple diff implementation
        diff = {
            "added": {},
            "removed": {},
            "changed": {},
        }

        # Get all keys
        all_keys = set(config1.keys()) | set(config2.keys())

        for key in all_keys:
            if key not in config1:
                diff["added"][key] = config2[key]
            elif key not in config2:
                diff["removed"][key] = config1[key]
            elif config1[key] != config2[key]:
                diff["changed"][key] = {
                    "old": config1[key],
                    "new": config2[key],
                }

        return diff

    def rollback_to_version(self, version: str) -> bool:
        """Rollback to specific version"""
        if version not in self.versions:
            return False

        # Set current version
        self.set_current_version(version)

        # Activate version
        self.activate_version(version)

        self.logger.info(f"Rolled back to version {version}")
        return True


# Global configuration version manager instance
_version_manager: Optional[ConfigVersionManager] = None


def get_version_manager() -> ConfigVersionManager:
    """Get global configuration version manager instance"""
    global _version_manager
    if _version_manager is None:
        _version_manager = ConfigVersionManager()
    return _version_manager


def set_version_manager(manager: ConfigVersionManager):
    """Set global configuration version manager instance"""
    global _version_manager
    _version_manager = manager
