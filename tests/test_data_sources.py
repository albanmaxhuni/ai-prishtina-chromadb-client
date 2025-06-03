"""
Unit tests for data source functionality in AIPrishtina VectorDB.
"""

import unittest
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np
from ai_prishtina_vectordb.data_sources import DataSource
from ai_prishtina_vectordb.logger import AIPrishtinaLogger

class TestDataSources(unittest.TestCase):
    """Test cases for data source functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = AIPrishtinaLogger(level="DEBUG")
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_text_source(self):
        """Test text data source."""
        # Create test file
        file_path = Path(self.temp_dir) / "test.txt"
        with open(file_path, "w") as f:
            f.write("Line 1\nLine 2\nLine 3")
            
        source = DataSource(source_type="text")
        data = source.load_data(file_path)
        
        self.assertIn("documents", data)
        self.assertIn("metadatas", data)
        self.assertIn("ids", data)
        self.assertEqual(len(data["documents"]), 3)
        
    def test_image_source(self):
        """Test image data source."""
        # Create dummy image data
        images = np.random.rand(2, 224, 224, 3)
        source = DataSource(source_type="image")
        data = source.load_data(images)
        
        self.assertIn("documents", data)
        self.assertIn("metadatas", data)
        self.assertIn("ids", data)
        self.assertEqual(len(data["documents"]), 2)
        
    def test_audio_source(self):
        """Test audio data source."""
        # Create dummy audio data
        audio = np.random.rand(2, 16000)
        source = DataSource(source_type="audio")
        data = source.load_data(audio)
        
        self.assertIn("documents", data)
        self.assertIn("metadatas", data)
        self.assertIn("ids", data)
        self.assertEqual(len(data["documents"]), 2)
        
    def test_video_source(self):
        """Test video data source."""
        # Create dummy video data
        video = np.random.rand(2, 30, 224, 224, 3)
        source = DataSource(source_type="video")
        data = source.load_data(video)
        
        self.assertIn("documents", data)
        self.assertIn("metadatas", data)
        self.assertIn("ids", data)
        self.assertEqual(len(data["documents"]), 2)
        
    def test_dataframe_source(self):
        """Test DataFrame data source."""
        df = pd.DataFrame({
            "text": ["A", "B", "C"],
            "metadata": [1, 2, 3]
        })
        
        source = DataSource(source_type="text")
        data = source.load_data(df, text_column="text")
        
        self.assertIn("documents", data)
        self.assertIn("metadatas", data)
        self.assertIn("ids", data)
        self.assertEqual(len(data["documents"]), 3)

if __name__ == '__main__':
    unittest.main() 