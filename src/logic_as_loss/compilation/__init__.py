"""Logic compilation strategies."""

from .base import LogicCompiler
from .tnorm_compiler import TNormCompiler

__all__ = [
    'LogicCompiler',
    'TNormCompiler',
]
