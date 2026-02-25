"""Deploy log summarization for a concise TUI deploy/warmup view."""

from __future__ import annotations

import re

from ..protocol.enums import BackendType, OperationType

_ERROR_WORD_RE = re.compile(
    r"\b(error|exception|traceback|fatal|failed|runtimeerror|assertionerror)\b",
    flags=re.IGNORECASE,
)
_OOM_RE = re.compile(
    r"(cuda out of memory|outofmemory|segmentation fault|\boom\b)",
    flags=re.IGNORECASE,
)
_HTTP_ERROR_CONTEXT_RE = re.compile(
    r"(\bHTTP\s*5\d\d\b|\b->\s*5\d\d\b|\b5\d\d\s+(Service Unavailable|Bad Gateway|Internal Server Error)\b)",
    flags=re.IGNORECASE,
)
_VLLM_INFO_LINE_RE = re.compile(r"^\((APIServer|EngineCore[^)]*) pid=\d+\)\s+INFO\b")
_LLAMACPP_DOWNLOAD_INFLIGHT_RE = re.compile(r"\binflight=(\d+)\b")
_LLAMACPP_DOWNLOAD_RATE_RE = re.compile(r"\bavg_rate=([0-9]+(?:\.[0-9]+)?)MiB/s\b")


class DeployLogSummarizer:
    """Convert raw backend logs into canonical milestone messages."""

    def __init__(self, backend: BackendType) -> None:
        self.backend = backend
        self.reset()

    def reset(self) -> None:
        self._operation: OperationType | None = None
        self._seen_milestones: set[str] = set()

    def transform(self, line: str, operation: OperationType | None) -> list[str]:
        text = (line or "").rstrip()
        if not text:
            return []

        if operation != self._operation:
            self._operation = operation
            self._seen_milestones.clear()

        preserved = self._preserve_common_line(text)
        if preserved is not None:
            return preserved

        if self.backend == BackendType.LLAMACPP and self._is_llamacpp_readiness_503_noise(text):
            return []

        mapped = self._map_backend_line(text)
        if mapped:
            return self._emit_once(mapped)

        if self._is_error_like(text):
            return [text]

        if self._is_noise(text):
            return []

        return []

    def _emit_once(self, milestone: str) -> list[str]:
        if milestone in self._seen_milestones:
            return []
        self._seen_milestones.add(milestone)
        return [milestone]

    def _preserve_common_line(self, text: str) -> list[str] | None:
        stripped = text.strip()
        if text.startswith("Running: "):
            return [text]
        if stripped.startswith("env: "):
            return [text]
        if text.startswith("Probing readiness at: "):
            out = self._emit_once("Waiting for readiness")
            out.append(text)
            return out
        if text.startswith("Waiting for GPU scheduling"):
            return [text]
        if text == "GPU allocated; continuing readiness probe.":
            return [text]
        if text == "Server is ready!":
            return [text]
        if text.startswith("Test command:"):
            return [text]
        if text.startswith("=== OpenAI-compatible"):
            return [text]
        if text == "=========================":
            return [text]
        if text == "OpenAI-compatible connection summary:":
            return [text]
        if text.startswith("OpenCode custom provider:"):
            return [text]
        if stripped.startswith("Base URL: "):
            return [text]
        if stripped.startswith("Model ID: "):
            return [text]
        if stripped.startswith("Provider ID: "):
            return [text]
        if stripped.startswith("Display name: "):
            return [text]
        if stripped.startswith("API key: "):
            return [text]
        if stripped == "Models:":
            return [text]
        if stripped.startswith("- ID: "):
            return [text]
        if stripped.startswith("Name: "):
            return [text]
        if stripped.startswith("Headers: "):
            return [text]
        if "Created web function serve =>" in text and "http" in text:
            if "-dev.modal.run" in text:
                return []
            return [text]
        if text.startswith("View Deployment: "):
            return [text]
        if "App deployed in " in text:
            return [text]
        if text.startswith("Warning: "):
            return [text]
        return None

    def _map_backend_line(self, text: str) -> str | None:
        if self.backend == BackendType.LLAMACPP:
            return self._map_llamacpp_line(text)
        return self._map_vllm_line(text)

    def _map_llamacpp_line(self, text: str) -> str | None:
        if "cache hit:" in text:
            return "Loading cached model"
        if "cache miss" in text:
            return "Downloading model"
        # The preload helper prints a "downloading ..." banner unconditionally
        # before calling snapshot_download(), even for cache hits. Only show the
        # canonical download milestone once we see progress output.
        if "download in progress..." in text:
            if self._llamacpp_progress_indicates_actual_transfer(text):
                return "Downloading model"
            return None
        if "found GGUF entries:" in text:
            return "Loading cached model"
        if "using model file:" in text:
            return "Loading cached model"
        if "starting llama-server:" in text:
            return "Starting server"
        if text.startswith("ggml_cuda_init:") or text.startswith("load_backend:"):
            return "Initializing CUDA"
        if (
            text.startswith("main: loading model")
            or text.startswith("srv    load_model:")
            or text.startswith("llama_model_loader: loaded meta data")
        ):
            return "Loading model metadata"
        if text.startswith("load_tensors: loading model tensors"):
            return "Loading model weights"
        if text.startswith("load_tensors: offloading") or text.startswith("load_tensors: offloaded"):
            return "Loading weights on GPU"
        return None

    def _map_vllm_line(self, text: str) -> str | None:
        if text.startswith("Starting vLLM command:"):
            return "Starting server"
        if "vLLM API server version" in text:
            return "Starting server"
        if "Resolved architecture:" in text:
            return "Loading model metadata"
        if "Starting to load model " in text:
            return "Loading model weights"
        if "Time spent downloading weights" in text:
            return "Downloading model"
        if text.startswith("Loading safetensors checkpoint shards:"):
            return "Loading model weights"
        if "Model loading took " in text:
            return "Loading weights on GPU"
        if "Capturing CUDA graphs (" in text:
            return "Capturing CUDA graphs"
        if "Available KV cache memory" in text or "GPU KV cache size:" in text:
            return "Initializing KV cache"
        if self._is_vllm_compile_line(text):
            return "Compiling kernels"
        return None

    @staticmethod
    def _is_vllm_compile_line(text: str) -> bool:
        if "[monitor.py:" in text and "torch.compile" in text:
            return True
        if "[backends.py:" not in text:
            return False
        return (
            "torch.compile" in text
            or "Dynamo bytecode transform time" in text
            or "compiled graph" in text
        )

    def _is_noise(self, text: str) -> bool:
        if self.backend == BackendType.LLAMACPP:
            return self._is_llamacpp_noise(text)
        return self._is_vllm_noise(text)

    @staticmethod
    def _is_llamacpp_noise(text: str) -> bool:
        stripped = text.strip()
        return (
            "download in progress..." in text
            or text.startswith("llama_model_loader: - kv")
            or text.startswith("llama_model_loader: - type")
            or text.startswith("print_info:")
            or text.startswith("load: ")
            or text.startswith("system info:")
            or text.startswith("system_info:")
            or text.startswith("srv  log_server_r:")
            or text.startswith("common_init_result:")
            or text.startswith("llama_params_fit")
            or text.startswith("llama_params_fit_impl:")
            or text.startswith("llama_model_load_from_file_impl:")
            or text.startswith("load_tensors:")
            or text.startswith("main: n_parallel")
            or text.startswith("build:")
            or text.startswith("Running without SSL")
            or text.startswith("init: using ")
            or text.startswith("start: binding port")
            or stripped == "Next steps:"
            or stripped.startswith("Use the exact URL from the `Created web function serve =>")
            or stripped.startswith("(Modal may truncate long labels and append a hash")
            or stripped.startswith("1) Deploy the server:")
            or stripped.startswith("2) Once deployed, curl the server")
            or stripped.startswith("3) Faster iteration")
            or stripped.startswith("4) Image cache behavior")
            or stripped.startswith("Default: reuse cached image layers")
            or stripped.startswith("Force fresh latest image pull/build:")
            or stripped.startswith("Example: LLAMA_CPP_IMAGE_NO_CACHE=true modal deploy")
            or stripped.startswith("modal run ")
            or stripped.startswith("modal deploy ")
            or stripped.startswith("curl -sS -X POST ")
            or stripped.startswith("✅ Weights cached in Modal Volume")
            or stripped.startswith("ℹ️ `--deploy` runs `modal deploy`")
            or stripped.startswith("🚀 Deploying...")
            or stripped.startswith("✅ Deploy triggered.")
        )

    @staticmethod
    def _is_vllm_noise(text: str) -> bool:
        if _VLLM_INFO_LINE_RE.match(text):
            return True
        if text.lstrip().startswith("vllm serve "):
            return True
        if text.startswith("Loading safetensors checkpoint shards:"):
            return True
        if "Capturing CUDA graphs (" in text:
            return True
        return False

    @staticmethod
    def _is_llamacpp_readiness_503_noise(text: str) -> bool:
        if "/v1/completions" not in text or "503" not in text:
            return False
        return text.startswith("srv  log_server_r:") or text.lstrip().startswith("POST /v1/completions")

    @staticmethod
    def _is_error_like(text: str) -> bool:
        if _ERROR_WORD_RE.search(text) or _OOM_RE.search(text):
            return True
        if _HTTP_ERROR_CONTEXT_RE.search(text):
            return True
        return False

    @staticmethod
    def _llamacpp_progress_indicates_actual_transfer(text: str) -> bool:
        inflight_match = _LLAMACPP_DOWNLOAD_INFLIGHT_RE.search(text)
        if inflight_match and int(inflight_match.group(1)) > 0:
            return True
        rate_match = _LLAMACPP_DOWNLOAD_RATE_RE.search(text)
        if rate_match:
            try:
                return float(rate_match.group(1)) > 0.0
            except ValueError:
                return False
        return False
