"""Buffer management for host and GPU tensors."""

from grinnder.buffer.host import HostBuffer
from grinnder.buffer.device import DeviceBuffer

__all__ = ["HostBuffer", "DeviceBuffer"]
