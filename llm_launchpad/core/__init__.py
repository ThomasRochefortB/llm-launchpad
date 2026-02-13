"""Core layer: business logic and orchestration.

This package owns configuration, Modal subprocess execution,
and high-level workflow coordination. It emits Protocol events
only and has zero UI dependencies.
"""

from .config import ConfigStore
from .backend import ModalBackend
from .modal_gpu import fetch_modal_gpu_types
from .orchestrator import Orchestrator

__all__ = ["ConfigStore", "ModalBackend", "Orchestrator", "fetch_modal_gpu_types"]
