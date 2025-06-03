"""
Unit tests for metrics functionality in AIPrishtina VectorDB.
"""

import unittest
import time
import numpy as np
from ai_prishtina_vectordb.metrics import MetricsCollector, PerformanceMonitor
from ai_prishtina_vectordb.logger import AIPrishtinaLogger

class TestMetrics(unittest.TestCase):
    """Test cases for metrics functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = AIPrishtinaLogger(level="DEBUG")
        self.metrics = MetricsCollector(logger=self.logger)
        self.monitor = PerformanceMonitor(logger=self.logger)
        
    def test_metrics_collection(self):
        """Test metrics collection."""
        # Test search metrics
        self.metrics.record_search(
            query="test query",
            n_results=5,
            response_time=0.1
        )
        
        metrics = self.metrics.get_metrics()
        self.assertIn("search_metrics", metrics)
        self.assertEqual(metrics["search_metrics"]["total_queries"], 1)
        
        # Test embedding metrics
        self.metrics.record_embedding(
            n_documents=10,
            embedding_time=0.2
        )
        
        metrics = self.metrics.get_metrics()
        self.assertIn("embedding_metrics", metrics)
        self.assertEqual(metrics["embedding_metrics"]["total_documents"], 10)
        
    def test_performance_monitoring(self):
        """Test performance monitoring."""
        with self.monitor.measure("test_operation"):
            time.sleep(0.1)
            
        metrics = self.monitor.get_metrics()
        self.assertIn("test_operation", metrics)
        self.assertGreater(metrics["test_operation"]["avg_time"], 0)
        
    def test_metrics_reset(self):
        """Test metrics reset."""
        self.metrics.record_search(
            query="test query",
            n_results=5,
            response_time=0.1
        )
        
        self.metrics.reset()
        metrics = self.metrics.get_metrics()
        self.assertEqual(metrics["search_metrics"]["total_queries"], 0)
        
    def test_performance_thresholds(self):
        """Test performance thresholds."""
        self.monitor.set_threshold("test_operation", 0.2)
        
        with self.monitor.measure("test_operation"):
            time.sleep(0.1)
            
        self.assertFalse(self.monitor.is_threshold_exceeded("test_operation"))
        
        with self.monitor.measure("test_operation"):
            time.sleep(0.3)
            
        self.assertTrue(self.monitor.is_threshold_exceeded("test_operation"))

if __name__ == '__main__':
    unittest.main() 