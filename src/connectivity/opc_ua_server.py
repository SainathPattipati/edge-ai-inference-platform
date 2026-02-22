"""OPC-UA server for publishing inference results to SCADA/MES systems."""

from typing import Dict, Any, Callable
from asyncio import Task
import asyncio
from asyncua import Server, ua
from asyncua.common import Node
import logging


class OPCUAServer:
    """OPC-UA server exposing edge inference results."""
    
    def __init__(
        self,
        endpoint: str = "opc.tcp://0.0.0.0:4840",
        server_name: str = "ManufacturingEdgeAI"
    ):
        """
        Initialize OPC-UA server.
        
        Args:
            endpoint: OPC-UA server endpoint URL
            server_name: Server display name
        """
        self.endpoint = endpoint
        self.server_name = server_name
        self.server = Server()
        self.inference_nodes: Dict[str, Node] = {}
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize and start OPC-UA server."""
        await self.server.init()
        self.server.set_endpoint(self.endpoint)
        self.server.set_server_name(self.server_name)
        
        # Create custom namespace
        namespace_uri = "http://manufacturing-edge-ai.dev"
        ns_idx = await self.server.register_namespace(namespace_uri)
        
        # Create root object
        objects = self.server.get_objects_node()
        inference_obj = await objects.add_object(ns_idx, "InferenceResults")
        
        self.inference_obj = inference_obj
        self.ns_idx = ns_idx
        
        # Enable historical access
        await self.server.write_attribute_value(
            ua.NodeId("HistoryServerCapabilities"),
            ua.DataValue(ua.Variant(True, ua.VariantType.Boolean))
        )
    
    async def add_model_result_node(
        self,
        model_name: str,
        confidence_threshold: float = 0.7
    ) -> None:
        """
        Create OPC-UA node for model inference results.
        
        Args:
            model_name: Name of inference model
            confidence_threshold: Alert threshold for confidence
        """
        # Create folder for model
        model_obj = await self.inference_obj.add_object(self.ns_idx, model_name)
        
        # Add variables
        nodes = {}
        nodes["prediction"] = await model_obj.add_variable(
            self.ns_idx, "Prediction", 0.0
        )
        nodes["confidence"] = await model_obj.add_variable(
            self.ns_idx, "Confidence", 0.0
        )
        nodes["latency_ms"] = await model_obj.add_variable(
            self.ns_idx, "LatencyMs", 0.0
        )
        nodes["last_update"] = await model_obj.add_variable(
            self.ns_idx, "LastUpdate", ""
        )
        nodes["status"] = await model_obj.add_variable(
            self.ns_idx, "Status", "idle"
        )
        
        # Make variables writable and historyable
        for node in nodes.values():
            await node.set_writable(True)
            # Set historical access
            await node.set_attribute(
                ua.AttributeIds.Historizing,
                ua.DataValue(ua.Variant(True, ua.VariantType.Boolean))
            )
        
        self.inference_nodes[model_name] = nodes
    
    async def publish_result(
        self,
        model_name: str,
        prediction: float,
        confidence: float,
        latency_ms: float
    ) -> None:
        """
        Publish inference result to OPC-UA.
        
        Args:
            model_name: Model name (must be registered node)
            prediction: Model output value
            confidence: Confidence score 0-1
            latency_ms: Inference latency in milliseconds
        """
        if model_name not in self.inference_nodes:
            self.logger.warning(f"Model {model_name} not registered")
            return
        
        nodes = self.inference_nodes[model_name]
        timestamp = asyncio.get_event_loop().time()
        
        try:
            await nodes["prediction"].write_value(prediction)
            await nodes["confidence"].write_value(confidence)
            await nodes["latency_ms"].write_value(latency_ms)
            await nodes["last_update"].write_value(str(timestamp))
            
            # Update status
            status = "alert" if confidence < 0.7 else "ok"
            await nodes["status"].write_value(status)
        
        except Exception as e:
            self.logger.error(f"Error publishing result: {e}")
    
    async def start(self) -> None:
        """Start OPC-UA server."""
        try:
            await self.server.start()
            self.logger.info(f"OPC-UA server started at {self.endpoint}")
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop OPC-UA server."""
        await self.server.stop()
        self.logger.info("OPC-UA server stopped")
    
    async def run_async(self) -> None:
        """Run server indefinitely."""
        await self.initialize()
        await self.start()
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.stop()
