"""Textual TUI frontend for llm-launchpad.

Consumes protocol events from the Core layer and renders
screens and widgets. No direct subprocess execution.
"""

from __future__ import annotations

from .compat import install as _install_textual_compat

_install_textual_compat()
