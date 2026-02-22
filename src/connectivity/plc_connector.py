"""
PLC connector supporting multiple industrial protocols.

Protocols: Modbus TCP/RTU, EtherNet/IP, PROFINET, Siemens S7
"""

from typing import Dict, List, Any
from enum import Enum
from dataclasses import dataclass
from pymodbus.client import ModbusTcpClient, ModbusSerialClient


class ProtocolType(str, Enum):
    """Supported industrial protocols."""
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    ETHERNET_IP = "ethernet_ip"
    PROFINET = "profinet"
    SIEMENS_S7 = "siemens_s7"


@dataclass
class RegisterMapping:
    """Mapping from register to process variable."""
    register_address: int
    name: str
    data_type: str  # "uint16", "int32", "float32"
    access: str  # "read" or "write"
    scale_factor: float = 1.0
    offset: float = 0.0


class PLCConnector:
    """Connect to industrial PLCs and I/O devices."""
    
    def __init__(
        self,
        protocol: ProtocolType,
        host: str | None = None,
        port: int | None = None,
        device_address: int = 1
    ):
        """
        Initialize PLC connector.
        
        Args:
            protocol: Communication protocol
            host: For TCP protocols
            port: For TCP protocols
            device_address: Modbus slave address
        """
        self.protocol = protocol
        self.host = host or "localhost"
        self.port = port or self._default_port(protocol)
        self.device_address = device_address
        self.client: Any = None
        self.register_map: Dict[str, RegisterMapping] = {}
        
        self._connect()
    
    def _default_port(self, protocol: ProtocolType) -> int:
        """Get default port for protocol."""
        ports = {
            ProtocolType.MODBUS_TCP: 502,
            ProtocolType.ETHERNET_IP: 44818,
            ProtocolType.PROFINET: 34962,
            ProtocolType.SIEMENS_S7: 102,
        }
        return ports.get(protocol, 0)
    
    def _connect(self) -> None:
        """Establish connection to PLC."""
        if self.protocol == ProtocolType.MODBUS_TCP:
            self.client = ModbusTcpClient(self.host, self.port)
            self.client.connect()
        elif self.protocol == ProtocolType.MODBUS_RTU:
            self.client = ModbusSerialClient(
                method="rtu",
                port=self.host,
                baudrate=9600,
                stopbits=1,
                bytesize=8,
                parity="N"
            )
            self.client.connect()
        else:
            raise NotImplementedError(f"Protocol {self.protocol} not yet implemented")
    
    def add_register_mapping(self, mapping: RegisterMapping) -> None:
        """Add variable to register mapping."""
        self.register_map[mapping.name] = mapping
    
    def read_register(self, var_name: str) -> float | None:
        """
        Read process variable from PLC.
        
        Args:
            var_name: Variable name from mapping
        
        Returns:
            Scaled value or None on error
        """
        if var_name not in self.register_map:
            return None
        
        mapping = self.register_map[var_name]
        
        if self.protocol in [ProtocolType.MODBUS_TCP, ProtocolType.MODBUS_RTU]:
            result = self.client.read_holding_registers(
                mapping.register_address,
                count=1,
                slave=self.device_address
            )
            
            if result.isError():
                return None
            
            raw_value = result.registers[0]
            # Apply scaling
            return raw_value * mapping.scale_factor + mapping.offset
        
        return None
    
    def write_register(self, var_name: str, value: float) -> bool:
        """
        Write setpoint to PLC.
        
        Args:
            var_name: Variable name
            value: Value to write
        
        Returns:
            Success status
        """
        if var_name not in self.register_map:
            return False
        
        mapping = self.register_map[var_name]
        
        if mapping.access != "write":
            return False
        
        # Reverse scaling
        raw_value = int((value - mapping.offset) / mapping.scale_factor)
        
        if self.protocol in [ProtocolType.MODBUS_TCP, ProtocolType.MODBUS_RTU]:
            result = self.client.write_register(
                mapping.register_address,
                raw_value,
                slave=self.device_address
            )
            return not result.isError()
        
        return False
    
    def read_multiple(self, var_names: List[str]) -> Dict[str, float]:
        """Read multiple variables efficiently."""
        results = {}
        for var_name in var_names:
            results[var_name] = self.read_register(var_name)
        return results
    
    def close(self) -> None:
        """Close PLC connection."""
        if self.client:
            self.client.close()
