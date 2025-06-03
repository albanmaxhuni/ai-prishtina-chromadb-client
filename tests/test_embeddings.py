"""
Unit tests for embedding functionality in AIPrishtina VectorDB.
"""

import unittest
import numpy as np
from ai_prishtina_vectordb.embeddings import EmbeddingModel

class TestEmbeddings(unittest.TestCase):
    """Test cases for embedding functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = EmbeddingModel()
        
    def test_text_embeddings(self):
        """Test text embedding generation."""
        texts = ["Welcome to AIPrishtina", "Test embedding"]
        embeddings = self.model.embed_text(texts)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(embeddings), len(texts))
        self.assertTrue(np.all(np.isfinite(embeddings)))
        
    def test_image_embeddings(self):
        """Test image embedding generation."""
        # Create dummy image data
        images = np.random.rand(2, 224, 224, 3)
        embeddings = self.model.embed_image(images)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(embeddings), len(images))
        self.assertTrue(np.all(np.isfinite(embeddings)))
        
    def test_audio_embeddings(self):
        """Test audio embedding generation."""
        # Create dummy audio data
        audio = np.random.rand(2, 16000)
        embeddings = self.model.embed_audio(audio)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(embeddings), len(audio))
        self.assertTrue(np.all(np.isfinite(embeddings)))
        
    def test_video_embeddings(self):
        """Test video embedding generation."""
        # Create dummy video data
        video = np.random.rand(2, 30, 224, 224, 3)
        embeddings = self.model.embed_video(video)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(embeddings), len(video))
        self.assertTrue(np.all(np.isfinite(embeddings)))
        
    def test_embedding_dimensions(self):
        """Test embedding dimensions."""
        texts = ["Test"]
        embeddings = self.model.embed_text(texts)
        
        self.assertEqual(embeddings.shape[1], self.model.dimension)
        
    def test_embedding_normalization(self):
        """Test embedding normalization."""
        texts = ["Test"]
        embeddings = self.model.embed_text(texts)
        
        # Check if embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-6))

if __name__ == '__main__':
    unittest.main() 