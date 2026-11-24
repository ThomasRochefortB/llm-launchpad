"""GGUF weight-file rules shared by Hub discovery and serving images."""

from __future__ import annotations

_AUXILIARY_GGUF_MARKERS = ("mmproj", "imatrix", "dspark", "draft", "eagle")
_AUXILIARY_GGUF_DIRS = ("mtp", "mtp-*", "engram")


def is_dspark_gguf(filename: str) -> bool:
    """Identify a GGUF explicitly named as a DSpark draft."""
    name = filename.casefold()
    return name.endswith(".gguf") and "dspark" in name


def is_engram_table_file(filename: str) -> bool:
    """Identify a standalone external Engram (--ple) GGUF table."""
    lowered = filename.strip().casefold()
    if not lowered.endswith((".gguf_file", ".gguf")):
        return False
    parts = lowered.split("/")
    return "engram" in parts[:-1] or "engram" in parts[-1]


def is_target_gguf(filename: str) -> bool:
    """Identify serving weights, excluding known GGUF add-ons."""
    lowered = filename.casefold()
    parts = lowered.split("/")
    return (
        lowered.endswith(".gguf")
        and not any(marker in lowered for marker in _AUXILIARY_GGUF_MARKERS)
        and not any(part in ("mtp", "engram") or part.startswith("mtp-") for part in parts[:-1])
    )


def auxiliary_gguf_ignore_patterns() -> list[str]:
    """Build case-insensitive Hugging Face glob exclusions for weight downloads."""
    def case_insensitive(pattern: str) -> str:
        return "".join(f"[{char}{char.upper()}]" if char.isalpha() else char for char in pattern)

    patterns = [f"*{marker}*" for marker in _AUXILIARY_GGUF_MARKERS]
    for directory in _AUXILIARY_GGUF_DIRS:
        patterns.extend((f"{directory}/*", f"*/{directory}/*"))
    return [case_insensitive(pattern) for pattern in patterns]
