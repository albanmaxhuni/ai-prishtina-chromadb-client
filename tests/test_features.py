"""Tests for feature extraction functionality."""

import pytest
import numpy as np
import os
import shutil
from ai_prishtina_vectordb.features import (
    FeatureConfig,
    FeatureExtractor,
    TextFeatureExtractor,
    FeatureProcessor,
    FeatureRegistry
)
from ai_prishtina_vectordb.exceptions import FeatureError
import unittest.mock

@pytest.fixture(scope="function")
def test_db_dir(tmp_path):
    """Create a temporary directory for the test database."""
    db_dir = tmp_path / "test_chroma_db"
    db_dir.mkdir(exist_ok=True)
    yield str(db_dir)
    # Cleanup after test
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

@pytest.fixture
def feature_config(test_db_dir):
    """Create a basic feature configuration."""
    return FeatureConfig(
        normalize=True,
        embedding_function="sentence_transformer",
        collection_name="test_collection",
        persist_directory=test_db_dir
    )

@pytest.fixture
def text_extractor(feature_config):
    """Create a text feature extractor."""
    return TextFeatureExtractor(feature_config)

@pytest.fixture
def feature_processor(feature_config):
    """Create a feature processor."""
    processor = FeatureProcessor(feature_config)
    yield processor
    # Cleanup after test
    if processor.client:
        processor.client.reset()

@pytest.fixture
def feature_registry():
    """Create a feature registry."""
    return FeatureRegistry()

def test_feature_config_defaults():
    """Test default configuration values."""
    config = FeatureConfig()
    assert config.normalize is True
    assert config.dimensionality_reduction is None
    assert config.feature_scaling is True
    assert config.cache_features is True
    assert config.batch_size == 100
    assert config.embedding_function == "default"
    assert config.collection_name is None
    assert config.metadata is None
    assert config.persist_directory is None
    assert config.collection_metadata is None
    assert config.hnsw_config is None
    assert config.distance_function == "cosine"

def test_text_feature_extraction(text_extractor):
    """Test text feature extraction."""
    text = "This is a test text"
    features = text_extractor.extract(text)
    
    assert isinstance(features, np.ndarray)
    assert len(features) > 0
    
    # Test caching
    cached_features = text_extractor.extract(text)
    assert np.array_equal(features, cached_features)

def test_batch_text_extraction(text_extractor):
    """Test batch text feature extraction."""
    texts = ["First text", "Second text", "Third text"]
    features = text_extractor.batch_extract(texts)
    
    assert isinstance(features, list)
    assert len(features) == len(texts)
    assert all(isinstance(f, np.ndarray) for f in features)

def test_feature_processor_collection_operations(feature_processor):
    """Test collection operations in feature processor."""
    # Ensure collection is properly initialized
    assert feature_processor.collection is not None, "Collection not initialized"
    
    # Test adding to collection
    data = {"text": "Test document"}
    feature_processor.add_to_collection(
        data=data,
        id="test1",
        metadata={"source": "test"},
        documents=["Test document"]
    )
    
    # Test querying collection
    results = feature_processor.query_collection(
        query_data={"text": "Test document"},
        n_results=1,
        where={"source": "test"}
    )
    assert len(results["ids"][0]) > 0
    
    # Test updating collection
    features = feature_processor.process({"text": "Updated document"})
    feature_processor.update_collection(
        ids=["test1"],
        embeddings=[features.tolist()],
        metadatas=[{"source": "updated"}],
        documents=["Updated document"]
    )
    
    # Test collection stats
    stats = feature_processor.get_collection_stats()
    assert stats["count"] > 0
    assert stats["name"] == "test_collection"
    
    # Test deleting from collection
    feature_processor.delete_from_collection(ids=["test1"])
    stats_after = feature_processor.get_collection_stats()
    assert stats_after["count"] == 0

def test_feature_processor_error_handling(feature_config):
    """Test error handling in feature processor."""
    # Mock ChromaDB client and Settings to avoid initialization errors
    with unittest.mock.patch("chromadb.Client"), \
         unittest.mock.patch("chromadb.PersistentClient"), \
         unittest.mock.patch("chromadb.Settings"):
        processor = FeatureProcessor(feature_config)
        # Test processing empty data
        with pytest.raises(FeatureError):
            processor.process({})
        # Test collection operations without collection
        with pytest.raises(FeatureError):
            FeatureProcessor(FeatureConfig())

def test_feature_registry(text_extractor, feature_config, feature_registry):
    """Test feature registry functionality."""
    # Mock ChromaDB client to avoid initialization errors
    with unittest.mock.patch("chromadb.Client"), unittest.mock.patch("chromadb.PersistentClient"):
        feature_processor = FeatureProcessor(feature_config)
        # Test registering components
        feature_registry.register_extractor("text", text_extractor)
        feature_registry.register_processor("default", feature_processor)
        # Test getting registered components
        assert feature_registry.get_extractor("text") == text_extractor
        assert feature_registry.get_processor("default") == feature_processor
        # Test getting non-existent components
        with pytest.raises(FeatureError):
            feature_registry.get_extractor("non_existent")
        with pytest.raises(FeatureError):
            feature_registry.get_processor("non_existent")

@pytest.mark.skip(reason="Dimensionality reduction test temporarily disabled")
def test_dimensionality_reduction(feature_processor):
    """Test dimensionality reduction in feature processor."""
    config = FeatureConfig(
        dimensionality_reduction=64,
        collection_name="reduced_collection",
        persist_directory="./test_chroma_db"
    )
    processor = FeatureProcessor(config)
    
    data = {"text": "This is a test text for dimensionality reduction"}
    features = processor.process(data)
    assert len(features) == 64

@pytest.mark.skip(reason="Feature scaling test temporarily disabled")
def test_feature_scaling(feature_processor):
    """Test feature scaling in feature processor."""
    config = FeatureConfig(
        feature_scaling=True,
        collection_name="scaled_collection",
        persist_directory="./test_chroma_db"
    )
    processor = FeatureProcessor(config)
    
    data = {"text": "Test text for scaling"}
    features = processor.process(data)
    assert np.all(features >= 0) and np.all(features <= 1) 