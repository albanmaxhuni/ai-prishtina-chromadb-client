"""
AIPrishtina VectorDB - A professional Python library for efficient data vectorization and vector database operations.
"""

from .api import *  # Re-export all API components

__version__ = "0.1.0"

# For backward compatibility
from .database import Database
from .data_sources import DataSource
from .vectorizer import Vectorizer
from .embeddings import EmbeddingModel
from .config import Config, DatabaseConfig, CacheConfig, LoggingConfig
from .exceptions import (
    AIPrishtinaError,
    ConfigurationError,
    DataSourceError,
    EmbeddingError,
    DatabaseError,
    ValidationError,
    CacheError,
    IndexError,
    SearchError,
    BatchProcessingError,
    ResourceNotFoundError,
    AuthenticationError,
    RateLimitError,
    FeatureError,
    QueryError,
    StorageError,
)
from .logger import AIPrishtinaLogger
from .metrics import MetricsCollector, PerformanceMonitor
from .validation import (
    validate_metadata,
    validate_documents,
    validate_embeddings,
    validate_query_params,
    validate_batch_params,
    validate_index_params
)

__all__ = [
    # Feature Extraction
    "FeatureConfig",
    "FeatureExtractor",
    "TextFeatureExtractor",
    "ImageFeatureExtractor",
    "AudioFeatureExtractor",
    "FeatureProcessor",
    "FeatureRegistry",
    
    # Core Components
    "Database",
    "DataSource",
    "EmbeddingModel",
    "Vectorizer",
    
    # Configuration
    "Config",
    "DatabaseConfig",
    "CacheConfig",
    "LoggingConfig",
    
    # Utilities
    "AIPrishtinaLogger",
    "MetricsCollector",
    "PerformanceMonitor",
    
    # Validation
    "validate_metadata",
    "validate_documents",
    "validate_embeddings",
    "validate_query_params",
    "validate_batch_params",
    "validate_index_params",
    
    # Exceptions
    "AIPrishtinaError",
    "ConfigurationError",
    "DataSourceError",
    "EmbeddingError",
    "DatabaseError",
    "ValidationError",
    "CacheError",
    "IndexError",
    "SearchError",
    "BatchProcessingError",
    "ResourceNotFoundError",
    "AuthenticationError",
    "RateLimitError",
    "FeatureError",
    "QueryError",
    "StorageError"
] 