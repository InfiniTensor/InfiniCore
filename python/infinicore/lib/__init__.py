"""
Internal Python package for the compiled InfiniCore extension.

The `_infinicore` extension module is built/installed into this package by:
  `xmake build _infinicore && xmake install _infinicore`
"""

from . import _infinicore

__all__ = ["_infinicore"]

