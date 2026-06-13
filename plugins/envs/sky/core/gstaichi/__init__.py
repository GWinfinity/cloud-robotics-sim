"""
gstaichi Shim - wraps taichi to provide gstaichi compatibility.

This module re-exports all public taichi APIs so the sky plugin's
custom Genesis fork can use it as a drop-in replacement for gstaichi.
"""
import taichi as _taichi

# Re-export all public names from taichi
__all__ = []

# Copy all public attributes from taichi
for _attr in dir(_taichi):
    if not _attr.startswith('_'):
        globals()[_attr] = getattr(_taichi, _attr)
        __all__.append(_attr)

# Explicitly ensure commonly-used internals are available
# (used by genesis's init and solvers)
_internal_modules = ['_logging', '_lib', '_kernels', 'lang']
for _mod_name in _internal_modules:
    if hasattr(_taichi, _mod_name):
        globals()[_mod_name] = getattr(_taichi, _mod_name)

# Ensure lang submodules are accessible (used by sky solvers)
try:
    from taichi import lang as _lang
    globals()['lang'] = _lang
except ImportError:
    pass

try:
    from taichi.lang import impl as _impl
    globals()['impl'] = _impl
except ImportError:
    pass

# taichi version compatibility
__version__ = _taichi.__version__
