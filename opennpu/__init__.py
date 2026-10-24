"""OpenNPU: run any model on AMD Ryzen AI (XDNA2) NPUs under Linux."""
__version__ = "0.3.0"
__all__ = ["offload", "parity", "get_engine", "NPUUnavailable", "NPULinear", "OffloadReport"]

# The offload API needs torch; the CLI, the server front end and the web UI don't, so they
# keep working (and start fast) in an environment without it.
try:
    from .engine import get_engine, NPUUnavailable
    from .offload import offload, parity, NPULinear, OffloadReport
except ImportError as _e:  # pragma: no cover - exercised only without torch
    if getattr(_e, "name", None) not in ("torch", "numpy"):
        raise
