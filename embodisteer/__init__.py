"""Public package boundary for the EmbodiSteer research codebase.

The bundled model/data namespaces remain importable for checkpoint
compatibility. New code should import policies and runtime metadata through
this package.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
