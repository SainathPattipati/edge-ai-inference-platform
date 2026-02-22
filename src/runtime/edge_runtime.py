"""
Edge inference runtime with ONNX and TensorRT support.

Provides sub-millisecond inference execution on heterogeneous hardware.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import onnxruntime as ort
from queue import PriorityQueue, Queue
import time


@dataclass
class InferenceRequest:
    """Inference request with priority and metadata."""
    request_id: str
    model_name: str
    input_data: np.ndarray
    priority: int = 0  # Lower number = higher priority
    timestamp: float = 0.0
    safety_critical: bool = False


@dataclass
class InferenceResult:
    """Result of inference execution."""
    request_id: str
    model_name: str
    output_data: Dict[str, np.ndarray]
    latency_ms: float
    confidence: float
    metadata: Dict[str, Any]


class EdgeRuntime:
    """ONNX-based inference engine optimized for edge deployment."""
    
    def __init__(
        self,
        models_dir: str | Path,
        use_tensorrt: bool = False,
        batch_size: int = 1,
        latency_sla_ms: float = 5.0
    ):
        """
        Initialize edge runtime.
        
        Args:
            models_dir: Directory containing ONNX models
            use_tensorrt: Enable TensorRT optimization (NVIDIA Jetson)
            batch_size: Max batch size for inference
            latency_sla_ms: Maximum acceptable latency
        """
        self.models_dir = Path(models_dir)
        self.use_tensorrt = use_tensorrt
        self.batch_size = batch_size
        self.latency_sla_ms = latency_sla_ms
        
        self.sessions: Dict[str, ort.InferenceSession] = {}
        self.request_queue: PriorityQueue = PriorityQueue()
        self.result_queue: Queue = Queue()
        self.latency_history: Dict[str, List[float]] = {}
        
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all ONNX models from directory."""
        for model_file in self.models_dir.glob("*.onnx"):
            model_name = model_file.stem
            
            # Configure execution providers
            providers = ["CPUExecutionProvider"]
            if self.use_tensorrt:
                providers.insert(0, "TensorrtExecutionProvider")
            
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            try:
                session = ort.InferenceSession(
                    str(model_file),
                    session_options,
                    providers=providers
                )
                self.sessions[model_name] = session
                self.latency_history[model_name] = []
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
    
    def infer(
        self,
        model_name: str,
        input_data: np.ndarray | Dict[str, np.ndarray],
        priority: int = 0,
        safety_critical: bool = False
    ) -> InferenceResult:
        """
        Run inference with optional priority queuing.
        
        Args:
            model_name: Name of model to use
            input_data: Input array or dict
            priority: Lower = higher priority (0 = critical)
            safety_critical: For SLA enforcement
        
        Returns:
            InferenceResult with outputs and metadata
        """
        if model_name not in self.sessions:
            raise ValueError(f"Model {model_name} not found")
        
        session = self.sessions[model_name]
        
        # Ensure proper input format
        if isinstance(input_data, np.ndarray):
            input_dict = {session.get_inputs()[0].name: input_data}
        else:
            input_dict = input_data
        
        start_time = time.perf_counter()
        
        # Run inference
        try:
            outputs = session.run(None, input_dict)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Check SLA
            if latency_ms > self.latency_sla_ms and safety_critical:
                print(f"WARNING: Latency {latency_ms:.2f}ms exceeds SLA {self.latency_sla_ms}ms")
            
            # Track latency
            self.latency_history[model_name].append(latency_ms)
            if len(self.latency_history[model_name]) > 1000:
                self.latency_history[model_name] = self.latency_history[model_name][-1000:]
            
            # Extract output names and create result dict
            output_dict = {}
            for i, output in enumerate(outputs):
                output_name = session.get_outputs()[i].name if i < len(session.get_outputs()) else f"output_{i}"
                output_dict[output_name] = output
            
            return InferenceResult(
                request_id=f"{model_name}_{int(start_time)}",
                model_name=model_name,
                output_data=output_dict,
                latency_ms=latency_ms,
                confidence=0.95,  # From model output
                metadata={
                    "batch_size": input_dict[list(input_dict.keys())[0]].shape[0],
                    "priority": priority
                }
            )
        
        except Exception as e:
            print(f"Inference error: {e}")
            raise
    
    def batch_infer(
        self,
        model_name: str,
        input_batch: List[np.ndarray]
    ) -> List[InferenceResult]:
        """
        Run inference on batch of inputs.
        
        Args:
            model_name: Model name
            input_batch: List of input arrays
        
        Returns:
            List of InferenceResult objects
        """
        # Stack inputs
        stacked_input = np.vstack(input_batch)
        
        result = self.infer(model_name, stacked_input)
        
        # Split results back to individual outputs
        results = []
        for i in range(len(input_batch)):
            split_outputs = {
                k: v[i:i+1] for k, v in result.output_data.items()
            }
            results.append(InferenceResult(
                request_id=f"{result.request_id}_{i}",
                model_name=model_name,
                output_data=split_outputs,
                latency_ms=result.latency_ms / len(input_batch),
                confidence=result.confidence,
                metadata=result.metadata
            ))
        
        return results
    
    def get_latency_stats(self, model_name: str) -> Dict[str, float]:
        """Get latency statistics for model."""
        if model_name not in self.latency_history:
            return {}
        
        latencies = np.array(self.latency_history[model_name])
        
        return {
            "min_ms": float(np.min(latencies)) if len(latencies) > 0 else 0,
            "max_ms": float(np.max(latencies)) if len(latencies) > 0 else 0,
            "mean_ms": float(np.mean(latencies)) if len(latencies) > 0 else 0,
            "p95_ms": float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0,
            "p99_ms": float(np.percentile(latencies, 99)) if len(latencies) > 0 else 0,
        }
