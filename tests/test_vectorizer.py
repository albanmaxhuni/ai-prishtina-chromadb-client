"""
Unit tests for the vectorizer module of AIPrishtina VectorDB.
"""

import unittest
import numpy as np
import pytest
from ai_prishtina_vectordb.vectorizer import Vectorizer
from ai_prishtina_vectordb.embeddings import EmbeddingModel

class TestVectorizer(unittest.TestCase):
    """Test cases for the Vectorizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vectorizer = Vectorizer()
        self.embedding_model = EmbeddingModel()
        
    @pytest.mark.asyncio
    async def test_vectorize_text(self):
        """Test text vectorization."""
        texts = ["Hello world", "Test vectorization"]
        vectors = await self.vectorizer.vectorize_text(texts)
        
        self.assertIsInstance(vectors, np.ndarray)
        self.assertEqual(len(vectors), len(texts))
        self.assertTrue(np.all(np.isfinite(vectors)))
        
    @pytest.mark.asyncio
    async def test_vectorize_numerical(self):
        """Test numerical data vectorization."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        vectors = await self.vectorizer.vectorize_numerical(data)
        
        self.assertIsInstance(vectors, np.ndarray)
        self.assertEqual(vectors.shape, data.shape)
        self.assertTrue(np.all(np.isfinite(vectors)))
        
    @pytest.mark.asyncio
    async def test_vectorize_categorical(self):
        """Test categorical data vectorization."""
        categories = ["A", "B", "C", "A"]
        vectors = await self.vectorizer.vectorize_categorical(categories)
        
        self.assertIsInstance(vectors, np.ndarray)
        self.assertEqual(len(vectors), len(categories))
        self.assertTrue(np.all(np.isfinite(vectors)))
        
    @pytest.mark.asyncio
    async def test_normalize_vectors(self):
        """Test vector normalization."""
        vectors = np.array([[1, 2, 3], [4, 5, 6]])
        normalized = await self.vectorizer._normalize_vectors(vectors)
        
        self.assertIsInstance(normalized, np.ndarray)
        self.assertEqual(normalized.shape, vectors.shape)
        self.assertTrue(np.all(np.isfinite(normalized)))
        
        # Check if vectors are normalized (unit length)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms))

if __name__ == '__main__':
    unittest.main() 