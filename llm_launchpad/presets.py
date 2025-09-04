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
    # Lighter coding preset examples (adjust to available repos in your HF account)
    # These are illustrative and may need to be updated to match available GGUF repos
    "qwen2.5-coder-7b": {
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "quant": "Q4_K_M",
        "revision": None,
    },
    "deepseek-coder-lite": {
        "repo_id": "TheBloke/deepseek-coder-6.7b-instruct-GGUF",
        "quant": "Q4_K_M",
        "revision": None,
    },
}


