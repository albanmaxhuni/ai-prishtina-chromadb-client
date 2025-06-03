"""
Unit tests for the database module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from ai_prishtina_vectordb import Database

@pytest.fixture
def db():
    """Create a test database instance."""
    return Database(
        collection_name="test_collection",
        config={
            "embedding_model": "all-MiniLM-L6-v2",
            "index_type": "default"
        }
    )

def test_database_initialization(db):
    """Test database initialization."""
    assert db.collection.name == "test_collection"
    assert db.collection is not None

def test_add_documents(db):
    """Test adding documents to the database."""
    documents = ["Hello world", "Welcome to AIPrishtina"]
    metadatas = [{"source": "test"}, {"source": "test"}]
    ids = ["1", "2"]
    
    # Add documents with precomputed embeddings to avoid model loading
    embeddings = np.random.rand(2, 384)  # Using 384 dimensions for MiniLM
    db.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    
    results = db.get(ids=ids)
    assert len(results["documents"]) == 2
    assert results["documents"] == documents
    assert results["metadatas"] == metadatas

def test_query_documents(db):
    """Test querying documents from the database."""
    # Add test documents with precomputed embeddings
    documents = ["Hello world", "Welcome to AIPrishtina"]
    metadatas = [{"source": "test"}, {"source": "test"}]
    ids = ["1", "2"]
    embeddings = np.random.rand(2, 384)
    db.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    
    # Query with precomputed embedding
    query_embedding = np.random.rand(1, 384)
    results = db.query(query_embeddings=query_embedding, n_results=1)
    assert len(results["documents"]) == 1

def test_delete_documents(db):
    """Test deleting documents from the database."""
    # Add test documents with precomputed embeddings
    documents = ["Hello world", "Welcome to AIPrishtina"]
    metadatas = [{"source": "test"}, {"source": "test"}]
    ids = ["1", "2"]
    embeddings = np.random.rand(2, 384)
    db.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    
    # Delete documents
    db.delete(ids=ids)
    
    # Verify deletion
    results = db.get(ids=ids)
    assert len(results["documents"]) == 0

def test_add_with_embeddings(db):
    """Test adding with precomputed embeddings."""
    docs = ["Doc 1", "Doc 2"]
    embeddings = np.random.rand(2, 384)
    metadatas = [{"source": "test"}, {"source": "test"}]
    ids = ["1", "2"]
    
    db.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)
    
    results = db.get(ids=ids)
    assert len(results["documents"]) == 2
    assert results["documents"] == docs
    assert results["metadatas"] == metadatas

def test_delete_and_get(db):
    """Test deleting and getting vectors."""
    docs = ["Doc A", "Doc B"]
    embeddings = np.random.rand(2, 384)
    metadatas = [{"cat": "A"}, {"cat": "B"}]
    ids = ["1", "2"]
    db.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)
    db.delete(ids=["1"])
    result = db.get(ids=["2"])
    assert "documents" in result
    assert result["documents"][0] == "Doc B"

def test_update_vectors(db):
    """Test updating vectors in the database."""
    docs = ["Old Doc"]
    embeddings = np.random.rand(1, 384)  # Pre-compute embeddings
    metadatas = [{"cat": "old"}]
    ids = ["1"]
    db.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)
    
    # Update with new embeddings
    new_embeddings = np.random.rand(1, 384)
    db.update(ids=ids, documents=["New Doc"], embeddings=new_embeddings, metadatas=[{"cat": "new"}])
    result = db.get(ids=ids)
    assert result["documents"][0] == "New Doc"

@pytest.mark.skip(reason="ChromaDB Collection does not support create_index")
def test_create_index(db):
    """Test creating an index."""
    db.create_index(
        index_type="hnsw",
        M=8,
        ef_construction=40,
        ef_search=50
    )

def setUp(self):
    """Set up test fixtures."""
    self.db = Database(
        collection_name="test_collection",
        config={
            "embedding_model": "all-MiniLM-L6-v2",
            "index_type": "default",
            "index_params": {}
        }
    )
    self.test_documents = [
        {"text": "Test document 1", "metadata": {"source": "test"}},
        {"text": "Test document 2", "metadata": {"source": "test"}}
    ]

def tearDown(self):
    """Clean up test fixtures."""
    try:
        self.db.collection.delete()
    except:
        pass 