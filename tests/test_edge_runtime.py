"""Unit tests for edge runtime."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.runtime.edge_runtime import EdgeRuntime, InferenceResult


class TestEdgeRuntime:
    """Test suite for EdgeRuntime."""
    
    @pytest.fixture
    def runtime(self, tmp_path):
        """Create runtime with temp models directory."""
        # Create dummy ONNX file
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        
        # Mock ONNX runtime
        with patch("onnxruntime.InferenceSession"):
            runtime = EdgeRuntime(str(model_dir), batch_size=4)
        
        return runtime
    
    def test_latency_tracking(self, runtime):
        """Test latency statistics tracking."""
        # Manually add latency history
        runtime.latency_history["test_model"] = [5.0, 4.8, 5.2, 4.9]
        
        stats = runtime.get_latency_stats("test_model")
        
        assert stats["min_ms"] == 4.8
        assert stats["max_ms"] == 5.2
        assert 4.9 < stats["mean_ms"] < 5.1
        assert stats["p99_ms"] > stats["p95_ms"]
    
    def test_model_not_found(self, runtime):
        """Test error handling for missing model."""
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        with pytest.raises(ValueError):
            runtime.infer("nonexistent_model", input_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
