"""
Unit tests for embedding functionality in AIPrishtina VectorDB.
"""

import unittest
import numpy as np
import pytest
from ai_prishtina_vectordb.embeddings import EmbeddingModel

class TestEmbeddings(unittest.TestCase):
    """Test cases for embedding functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = EmbeddingModel()
        
    @pytest.mark.asyncio
    async def test_text_embeddings(self):
        """Test text embedding generation."""
        texts = ["Hello world", "Test embedding"]
        embeddings = await self.model.embed_text(texts)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(embeddings), len(texts))
        self.assertTrue(np.all(np.isfinite(embeddings)))
        
    @pytest.mark.asyncio
    async def test_image_embeddings(self):
        """Test image embedding generation."""
        # Create a dummy image array
        image = np.random.rand(224, 224, 3)
        embedding = await self.model.embed_image(image)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertTrue(np.all(np.isfinite(embedding)))
        
    @pytest.mark.asyncio
    async def test_audio_embeddings(self):
        """Test audio embedding generation."""
        # Create a dummy audio array
        audio = np.random.rand(16000)  # 1 second of audio at 16kHz
        embedding = await self.model.embed_audio(audio)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertTrue(np.all(np.isfinite(embedding)))
        
    @pytest.mark.asyncio
    async def test_video_embeddings(self):
        """Test video embedding generation."""
        # Create a dummy video array
        video = np.random.rand(30, 224, 224, 3)  # 30 frames
        embedding = await self.model.embed_video(video)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertTrue(np.all(np.isfinite(embedding)))
        
    @pytest.mark.asyncio
    async def test_embedding_dimensions(self):
        """Test embedding dimensions."""
        text = "Test embedding dimensions"
        embedding = await self.model.embed_text([text])
        
        self.assertEqual(len(embedding.shape), 2)
        self.assertEqual(embedding.shape[0], 1)  # Batch size
        self.assertTrue(embedding.shape[1] > 0)  # Embedding dimension
        
    @pytest.mark.asyncio
    async def test_embedding_normalization(self):
        """Test embedding normalization."""
        text = "Test embedding normalization"
        embedding = await self.model.embed_text([text])
        
        norm = np.linalg.norm(embedding, axis=1)
        np.testing.assert_array_almost_equal(norm, np.ones_like(norm))

if __name__ == '__main__':
    unittest.main() 