"""
Unit tests for metrics functionality in AIPrishtina VectorDB.
"""

import unittest
import pytest
import asyncio
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
        
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection."""
        # Test search metrics
        await self.metrics.record_search(
            query="test query",
            n_results=5,
            response_time=0.1
        )
        
        metrics = await self.metrics.get_metrics()
        self.assertIn("search_metrics", metrics)
        self.assertEqual(metrics["search_metrics"]["total_queries"], 1)
        
        # Test embedding metrics
        await self.metrics.record_embedding(
            n_documents=10,
            embedding_time=0.2
        )
        
        metrics = await self.metrics.get_metrics()
        self.assertIn("embedding_metrics", metrics)
        self.assertEqual(metrics["embedding_metrics"]["total_documents"], 10)
        
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance monitoring."""
        async with self.monitor.measure("test_operation"):
            await asyncio.sleep(0.1)
            
        metrics = await self.monitor.get_metrics()
        self.assertIn("test_operation", metrics)
        self.assertGreater(metrics["test_operation"]["avg_time"], 0)
        
    @pytest.mark.asyncio
    async def test_metrics_reset(self):
        """Test metrics reset."""
        # Record some metrics
        await self.metrics.record_search(
            query="test query",
            n_results=5,
            response_time=0.1
        )
        
        # Reset metrics
        await self.metrics.reset()
        
        # Check if metrics are reset
        metrics = await self.metrics.get_metrics()
        self.assertEqual(metrics["search_metrics"]["total_queries"], 0)
        
    @pytest.mark.asyncio
    async def test_performance_thresholds(self):
        """Test performance thresholds."""
        await self.monitor.set_threshold("test_operation", 0.2)
        
        async with self.monitor.measure("test_operation"):
            await asyncio.sleep(0.1)
            
        metrics = await self.monitor.get_metrics()
        self.assertIn("test_operation", metrics)
        self.assertLess(metrics["test_operation"]["avg_time"], 0.2)

if __name__ == '__main__':
    unittest.main() 