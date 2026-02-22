"""Example: Deploy defect detection model to edge device."""

from src.runtime.edge_runtime import EdgeRuntime, InferenceRequest
import numpy as np


def example_defect_detection():
    """Deploy and run defect detector at edge."""
    
    # Initialize runtime
    runtime = EdgeRuntime(
        models_dir="./models",
        use_tensorrt=False,  # Set True for NVIDIA Jetson
        batch_size=4,
        latency_sla_ms=5.0
    )
    
    # Simulate inspection camera data (224x224 RGB image)
    sample_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Run inference
    result = runtime.infer(
        model_name="defect_detector",
        input_data=sample_image,
        priority=0,
        safety_critical=True
    )
    
    print(f"Defect Detection Result:")
    print(f"  Prediction: {result.output_data}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Latency: {result.latency_ms:.2f}ms")
    
    # Get latency statistics
    stats = runtime.get_latency_stats("defect_detector")
    print(f"\nLatency Stats:")
    print(f"  Mean: {stats.get('mean_ms', 0):.2f}ms")
    print(f"  P99: {stats.get('p99_ms', 0):.2f}ms")


if __name__ == "__main__":
    example_defect_detection()
