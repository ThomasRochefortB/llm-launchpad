from typing import Dict, Any

"""Model presets for the llama.cpp Modal server.

These presets are convenience shortcuts for common GGUF models. You can always
pass your own --repo-id / --quant via the CLI entrypoint instead of using a preset.
"""

__all__ = ["PRESETS"]


PRESETS: Dict[str, Dict[str, Any]] = {
    # Heavy coding preset (default in this repo)
    "qwen3-coder-480b": {
        "repo_id": "unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF",
        "quant": "Q4_K_M",
        "revision": None,
    },
    # Mid-size coding preset
    "qwen3-coder-30b": {
        "repo_id": "unsloth/Qwen3-Coder-30B-A3B-Instruct-1M-GGUF",
        "quant": "Q4_K_M",
        "revision": None,
    },
    # General-purpose 20B model
    "gpt-oss-20b": {
        "repo_id": "unsloth/gpt-oss-20b-GGUF",
        "quant": "Q4_K_M",
        "revision": None,
    },
}


