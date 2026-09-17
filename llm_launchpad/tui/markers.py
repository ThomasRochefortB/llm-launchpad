"""Single-cell glyphs that mark a provider wherever it is named.

Every provider marker must occupy exactly one terminal cell. The Hugging Face
emoji is East Asian Width "W" (two cells) and the diamonds that used to mark
Prime Intellect and Artificial Analysis are "Ambiguous" -- Rich measures them
as one cell while a CJK-configured terminal draws two. Either way the row
slips out of its column, and in the ambiguous case Rich and the terminal
disagree about where the rest of the line begins. These five are all width
"N": unambiguous, single-cell, and visually distinct from one another.

They live here rather than beside one panel because the auth block and the
billing panel mark the same providers, and a marker that differs between them
reads as two different things.
"""

from __future__ import annotations

from ..protocol.enums import ComputeProvider

MODAL_MARKER = "▰"
PRIME_MARKER = "✦"
VAST_MARKER = "❖"
HUGGINGFACE_MARKER = "◉"
ARTIFICIAL_ANALYSIS_MARKER = "✱"

PROVIDER_MARKERS: dict[ComputeProvider, str] = {
    ComputeProvider.MODAL: MODAL_MARKER,
    ComputeProvider.PRIME: PRIME_MARKER,
    ComputeProvider.VAST: VAST_MARKER,
}


def provider_marker(provider: ComputeProvider) -> str:
    """Return the one-cell marker for ``provider``."""
    return PROVIDER_MARKERS[provider]
