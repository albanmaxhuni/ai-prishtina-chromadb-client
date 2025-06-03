"""
Configuration management for the AIPrishtina VectorDB library.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
import json
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Configuration for database settings."""
    collection_name: str = "ai_prishtina_collection"
    persist_directory: str = field(default_factory=lambda: os.path.join(os.getcwd(), ".chroma"))
    embedding_model: Optional[str] = None
    index_type: str = "hnsw"
    index_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheConfig:
    """Configuration for caching settings."""
    enabled: bool = True
    cache_type: str = "memory"  # memory, redis, file
    cache_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), ".cache"))
    max_size: int = 1000
    ttl: int = 3600  # Time to live in seconds
    redis_url: Optional[str] = None

@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    level: str = "INFO"
    log_file: str = field(default_factory=lambda: os.path.join(os.getcwd(), "logs", "ai_prishtina.log"))
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class Config:
    """Main configuration class for AIPrishtina VectorDB."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from a JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_file(self, config_path: str) -> None:
        """Save configuration to a JSON file."""
        config_dir = os.path.dirname(config_path)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def validate(self) -> None:
        """Validate the configuration settings."""
        # Ensure directories exist
        os.makedirs(self.database.persist_directory, exist_ok=True)
        os.makedirs(self.cache.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.logging.log_file), exist_ok=True)

        # Validate database settings
        if not self.database.collection_name:
            raise ValueError("Collection name cannot be empty")
        if self.database.index_type not in ["hnsw", "flat", "ivf"]:
            raise ValueError("Invalid index type")

        # Validate cache settings
        if self.cache.enabled:
            if self.cache.cache_type not in ["memory", "redis", "file"]:
                raise ValueError("Invalid cache type")
            if self.cache.cache_type == "redis" and not self.cache.redis_url:
                raise ValueError("Redis URL required for Redis cache")

        # Validate logging settings
        if self.logging.level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid logging level")

    def create_directories(self):
        """Create necessary directories."""
        Path(self.database.persist_directory).mkdir(parents=True, exist_ok=True)
        Path(self.cache.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.logging.log_file)).mkdir(parents=True, exist_ok=True) 