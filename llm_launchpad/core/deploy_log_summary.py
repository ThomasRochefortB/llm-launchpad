"""Turn raw deploy/warmup logs into a short, user-facing progress stream."""

from __future__ import annotations

import re

from ..protocol.enums import BackendType, OperationType

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
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
_LLAMACPP_DOWNLOAD_PCT_RE = re.compile(r"\bpct=(\d+)%")
_LLAMACPP_DOWNLOAD_TOTAL_RE = re.compile(r"/([0-9]+(?:\.[0-9]+)?)GiB")
_HF_FETCH_PCT_RE = re.compile(r"Fetching \d+ files:\s+(\d+)%")
_BEARER_RE = re.compile(r"(Authorization:\s*Bearer\s+)\S+", flags=re.IGNORECASE)
_PRIME_OFFER_RE = re.compile(
    r"Selected (?:Prime|provider) offer(?: \S+)?:\s*(\d+)\s*x\s+(\S+)(?: via \S+)?\s*\(([^)]+)\)",
    flags=re.IGNORECASE,
)
_COMMAND_RE = re.compile(
    r"^(Running:\s*)?(modal|uv|python|docker)\s",
    flags=re.IGNORECASE,
)

_DONE_MILESTONES = {
    "Server is ready!",
    "Endpoint published",
    "Model cached",
    "Machine ready",
    "Runtime ready",
    "Secure endpoint connected",
    "GPU allocated",
}

_DONE_PREFIXES = (
    "GPU ready:",
    "Done (",
    "Operation complete",
)
_STICKY_MILESTONES = {
    "Waiting for readiness",
    "Server is ready!",
}
_VLLM_SHARD_PCT_RE = re.compile(
    r"Loading safetensors checkpoint shards:\s+(\d+)%"
)
_PERCENT_IN_PARENS_RE = re.compile(r"\((\d+)%\)")
_PROGRESS_SUFFIX_RE = re.compile(r"^(?P<label>.*?)(?: \((?P<pct>\d+)%\))$")
_SUMMARY_MARKERS = ("✓ ", "· ", "✗ ")
_INFO_PREFIXES = (
    "Warning:",
    "Prime cache disk",
    "Tip:",
    "Press ",
    "Detail:",
    "Keeping failed",
    "Terminated failed",
)
SUMMARY_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def strip_ansi(text: str) -> str:
    """Remove ANSI color sequences from a log line."""
    return _ANSI_ESCAPE_RE.sub("", text or "")


def redact_log_secrets(text: str) -> str:
    """Replace bearer tokens so log scrollback cannot leak endpoint keys."""
    return _BEARER_RE.sub(r"\1$LLM_LAUNCHPAD_API_KEY", text or "")


def is_error_like(text: str) -> bool:
    """Return whether a line should surface as a failure rather than noise."""
    if _ERROR_WORD_RE.search(text) or _OOM_RE.search(text):
        return True
    return bool(_HTTP_ERROR_CONTEXT_RE.search(text))


def strip_summary_marker(text: str) -> str:
    """Remove a compact status marker so progress lines can be compared."""
    stripped = (text or "").rstrip()
    if stripped.startswith(_SUMMARY_MARKERS):
        return stripped[2:]
    if (
        len(stripped) >= 2
        and stripped[0] in SUMMARY_SPINNER_FRAMES
        and stripped[1] == " "
    ):
        return stripped[2:]
    return stripped


def summary_progress_parts(text: str) -> tuple[str, int | None]:
    """Split ``Downloading model (22%)`` into a stable label and percent."""
    stripped = strip_summary_marker(text).strip()
    match = _PROGRESS_SUFFIX_RE.fullmatch(stripped)
    if match:
        return match.group("label"), int(match.group("pct"))
    return stripped, None


def classify_summary_kind(text: str) -> str:
    """Classify a summary line as step, info, done, error, or blank."""
    stripped = strip_summary_marker(text).strip()
    if not stripped:
        return "blank"
    if is_error_like(stripped) or stripped.startswith("Error:"):
        return "error"
    if stripped.startswith(_INFO_PREFIXES):
        return "info"
    if stripped in _DONE_MILESTONES or stripped.startswith(_DONE_PREFIXES):
        return "done"
    return "step"


def percent_in_text(text: str) -> int | None:
    """Return a ``(N%)`` value when the line carries explicit progress."""
    match = _PERCENT_IN_PARENS_RE.search(text or "")
    if match is None:
        return None
    try:
        return max(0, min(100, int(match.group(1))))
    except ValueError:
        return None


def beautify_summary_line(text: str, *, spinner_frame: str | None = None) -> str:
    """Prefix a canonical milestone with a compact status marker."""
    stripped = (text or "").rstrip()
    if not stripped:
        return stripped
    if stripped.startswith(_SUMMARY_MARKERS) or (
        len(stripped) >= 2
        and stripped[0] in SUMMARY_SPINNER_FRAMES
        and stripped[1] == " "
    ):
        return stripped
    if is_error_like(stripped):
        return f"✗ {stripped}"
    if stripped in _DONE_MILESTONES or stripped.startswith(_DONE_PREFIXES):
        return f"✓ {stripped}"
    if spinner_frame:
        frame = spinner_frame[0]
        return f"{frame} {stripped}"
    return f"· {stripped}"


class DeployLogSummarizer:
    """Convert raw backend logs into canonical milestone messages."""

    def __init__(self, backend: BackendType) -> None:
        self.backend = backend
        self.reset()

    def reset(self) -> None:
        self._operation: OperationType | None = None
        self._seen_milestones: set[str] = set()

    def transform(self, line: str, operation: OperationType | None) -> list[str]:
        text = redact_log_secrets(strip_ansi(line or "")).rstrip()
        if not text:
            return []

        if operation != self._operation:
            self._operation = operation
            self._seen_milestones = {
                item for item in self._seen_milestones if item in _STICKY_MILESTONES
            }

        preserved = self._preserve_common_line(text)
        if preserved is not None:
            return preserved

        if self.backend == BackendType.LLAMACPP and self._is_llamacpp_readiness_503_noise(text):
            return []

        mapped = self._map_line(text)
        if mapped:
            return self._emit_once(mapped)

        if is_error_like(text):
            return [text]

        if self._is_noise(text):
            return []

        return []

    def transform_state(self, detail: str, operation: OperationType | None) -> list[str]:
        """Map orchestrator state-change details to the same milestone stream."""
        text = redact_log_secrets(strip_ansi(detail or "")).strip()
        if not text:
            return []

        if operation != self._operation:
            self._operation = operation
            self._seen_milestones = {
                item for item in self._seen_milestones if item in _STICKY_MILESTONES
            }

        mapped = self._map_state_detail(text)
        if mapped:
            return self._emit_once(mapped)
        if self._looks_like_command(text):
            return []
        if self._is_noise(text):
            return []
        return self._emit_once(text)

    @staticmethod
    def _download_milestone(pct: int | None) -> str:
        if pct is not None and pct > 0:
            return f"Downloading model ({pct}%)"
        return "Downloading model"

    def _emit_once(self, milestone: str) -> list[str]:
        if milestone in self._seen_milestones:
            return []
        self._seen_milestones.add(milestone)
        return [milestone]

    def _preserve_common_line(self, text: str) -> list[str] | None:
        stripped = text.strip()
        if text.startswith("Running: "):
            mapped = self._map_command_line(text[len("Running: ") :])
            return self._emit_once(mapped) if mapped else []
        if stripped.startswith("env: "):
            return []
        if text.startswith("Probing readiness at: ") or text.startswith("Probing "):
            return self._emit_once("Waiting for readiness")
        if text.startswith("Waiting for GPU scheduling"):
            return [text]
        if text in {"GPU allocated; continuing readiness probe.", "GPU allocated; continuing readiness probe"}:
            return self._emit_once("GPU allocated")
        if text == "Server is ready!":
            return [text]
        if text.startswith("Test command:"):
            return []
        if text.startswith("=== OpenAI-compatible") or text == "=========================":
            return []
        if text.startswith("OpenCode "):
            return []
        if stripped.startswith(
            (
                "Base URL: ",
                "Model ID: ",
                "Provider ID: ",
                "Display name: ",
                "API key: ",
                "Name: ",
                "Headers: ",
                "- ID: ",
            )
        ) or stripped == "Models:":
            return []
        if "Created web function serve" in text and "http" in text:
            return []
        if text.startswith("View Deployment: "):
            return []
        if "App deployed in " in text:
            return self._emit_once("Endpoint published")
        if text.startswith("Warning: "):
            return [text]
        return None

    def _map_line(self, text: str) -> str | None:
        mapped = self._map_prime_line(text)
        if mapped is not None:
            return mapped or None
        mapped = self._map_modal_cli_line(text)
        if mapped:
            return mapped
        if _HF_FETCH_PCT_RE.search(text) or (
            "Fetching " in text and " files:" in text
        ):
            return self._download_milestone(self._hf_fetch_percent(text))
        if self.backend == BackendType.LLAMACPP:
            return self._map_llamacpp_line(text)
        return self._map_vllm_line(text)

    def _map_state_detail(self, text: str) -> str | None:
        mapped = self._map_command_line(text)
        if mapped:
            return mapped
        mapped = self._map_prime_line(text)
        if mapped is not None:
            return mapped or None
        lowered = text.casefold()
        if "fetching prime gpu" in lowered:
            return "Finding a GPU"
        if lowered.startswith("provisioning prime pod") or lowered.startswith("provisioning"):
            return "Provisioning machine"
        if "installing the portable prime" in lowered or "installing runtime" in lowered:
            return "Installing runtime"
        if "creating a secure prime tunnel" in lowered:
            return "Opening secure endpoint"
        if "waiting for the public inference endpoint" in lowered:
            return "Waiting for readiness"
        if lowered.startswith("probing "):
            return "Waiting for readiness"
        if "preparing deployment" in lowered:
            return "Preparing deployment"
        return None

    def _map_command_line(self, text: str) -> str | None:
        stripped = text.strip()
        if "modal run" in stripped:
            return "Preparing model cache"
        if "modal deploy" in stripped:
            return "Publishing endpoint"
        return None

    @staticmethod
    def _looks_like_command(text: str) -> bool:
        stripped = text.strip()
        if _COMMAND_RE.match(stripped):
            return True
        return "llm_launchpad.backends" in stripped

    def _map_prime_line(self, text: str) -> str | None:
        """Return a friendly Prime milestone, or empty string to hide the line."""
        offer = _PRIME_OFFER_RE.search(text)
        if offer:
            count, gpu, extra = offer.group(1), offer.group(2), offer.group(3)
            gpu_label = gpu.replace("_", " ")
            extras = [part.strip() for part in extra.split(",") if part.strip()]
            detail = " · ".join([f"{count}× {gpu_label}", *extras])
            return f"GPU ready: {detail}"
        if "Selected provider offer:" in text:
            rest = text.split(":", 1)[-1].strip().replace("x ", "× ")
            return f"GPU ready: {rest}"

        if text.startswith("Prime runtime: portable bootstrap"):
            return ""
        if text.startswith("Prime pod created:"):
            return "Provisioning machine"
        if text.startswith("Prime pod state:"):
            state = text.split(":", 1)[-1].strip().split("/", 1)[0].strip().upper()
            if state in {"ACTIVE", "RUNNING"}:
                return "Machine ready"
            return "Provisioning machine"
        if text.startswith("Prime networking:"):
            return "Opening secure endpoint"
        if text.startswith("Prime Tunnel:"):
            lowered = text.casefold()
            if "connected" in lowered or "success" in lowered:
                return "Secure endpoint connected"
            return "Opening secure endpoint"
        if text.startswith("Prime runtime:"):
            detail = text.split(":", 1)[-1].strip().casefold()
            if "openai-compatible endpoint is ready" in detail:
                return "Runtime ready"
            if "downloading the model" in detail:
                pct = percent_in_text(text)
                if pct is not None:
                    return f"Downloading model ({pct}%)"
                return "Downloading model"
            if "loading the model" in detail:
                pct = percent_in_text(text)
                if pct is not None:
                    return f"Loading model ({pct}%)"
                return "Loading model"
            if "pulling" in detail:
                return "Pulling container image"
            return "Installing runtime"
        if text.startswith("Prime endpoint URL ready:") or text.startswith("Prime endpoint:"):
            return ""
        if text.startswith("Attaching Prime disk") or text.startswith("Reusing Prime cache disk"):
            return "Using cache disk"
        if text.startswith("Created Prime cache disk"):
            return "Created cache disk"
        if text.startswith("Terminated failed Prime pod"):
            return "Terminated failed machine"
        if text.startswith("Keeping failed Prime pod"):
            return "Keeping failed machine; billing may continue"
        if text.startswith("Prime cache disk"):
            return text
        if text.startswith("Selected Prime offer"):
            return "GPU ready"
        return None

    def _map_modal_cli_line(self, text: str) -> str | None:
        stripped = text.strip()
        if stripped.startswith("✓ Created objects") or stripped.startswith("Created objects"):
            return None
        if "App deployed in " in text:
            return "Endpoint published"
        return None

    def _map_llamacpp_line(self, text: str) -> str | None:
        if "cache hit:" in text:
            return "Loading cached model"
        if "cache miss" in text or "acquired download lease" in text:
            return "Downloading model"
        if "forcing fresh GGUF download" in text:
            return "Downloading model"
        if "download in progress..." in text:
            if self._llamacpp_progress_indicates_actual_transfer(text):
                return self._download_milestone(self._llamacpp_progress_percent(text))
            return None
        if "Weights cached in Modal Volume" in text:
            return "Model cached"
        if "found GGUF entries:" in text:
            return None
        if "using model file:" in text:
            return "Loading cached model"
        if "starting llama-server:" in text:
            return "Starting server"
        if text.startswith("ggml_cuda_init:") or text.startswith("load_backend:"):
            return "Initializing GPU"
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
        if text.startswith("common_init_from_params: warming up the model"):
            return "Warming up model"
        if text.startswith("main: model loaded"):
            return "Model loaded"
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
        shard_pct = _VLLM_SHARD_PCT_RE.search(text)
        if shard_pct:
            return f"Loading model weights ({int(shard_pct.group(1))}%)"
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
        stripped = text.strip()
        if stripped.startswith(("├", "└", "│", "╭", "╰")):
            return True
        if stripped.startswith("✓ Initialized") or stripped.startswith("✓ Created"):
            return True
        if stripped.startswith("✓ App completed") or stripped.startswith("Stopping app"):
            return True
        if "View run at" in text:
            return True
        if stripped.startswith("Fetching "):
            return True
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
            or text.startswith("llama_context:")
            or text.startswith("llama_kv_cache:")
            or text.startswith("sched_reserve:")
            or text.startswith("slot   load_model:")
            or text.startswith("load_tensors:")
            or text.startswith("main: n_parallel")
            or text.startswith("main: starting the main loop")
            or text.startswith("main: server is listening")
            or text.startswith("build:")
            or text.startswith("Running without SSL")
            or text.startswith("init: using ")
            or text.startswith("init: chat template")
            or text.startswith("start: binding port")
            or text.startswith("srv          init:")
            or text.startswith("srv  update_slots:")
            or text.startswith("srv    load_model: prompt cache")
            or text.startswith("srv    load_model: use `--cache-ram")
            or text.startswith("srv    load_model: for more info")
            or text.startswith("srv    load_model: initializing slots")
            or text.startswith("no implementations specified")
            or text.startswith("warn:")
            or stripped.startswith("📝 Saved config")
            or stripped.startswith("{")
            or stripped.startswith('"')
            or stripped == "}"
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
            or stripped.startswith("curl -s -X POST ")
            or stripped.startswith("✅ Weights cached in Modal Volume")
            or stripped.startswith("ℹ️ `--deploy` runs `modal deploy`")
            or stripped.startswith("🚀 Deploying...")
            or stripped.startswith("✅ Deploy triggered.")
            or "using llama-server" in text
        )

    @staticmethod
    def _is_vllm_noise(text: str) -> bool:
        if _VLLM_INFO_LINE_RE.match(text):
            return True
        if text.lstrip().startswith("vllm serve "):
            return True
        if text.startswith("Loading safetensors checkpoint shards:"):
            return True
        return "Capturing CUDA graphs (" in text

    @staticmethod
    def _is_llamacpp_readiness_503_noise(text: str) -> bool:
        if "/v1/completions" not in text or "503" not in text:
            return False
        return text.startswith("srv  log_server_r:") or text.lstrip().startswith("POST /v1/completions")

    @staticmethod
    def _llamacpp_progress_indicates_actual_transfer(text: str) -> bool:
        inflight_match = _LLAMACPP_DOWNLOAD_INFLIGHT_RE.search(text)
        if inflight_match and int(inflight_match.group(1)) > 0:
            return True
        rate_match = _LLAMACPP_DOWNLOAD_RATE_RE.search(text)
        if rate_match:
            try:
                if float(rate_match.group(1)) > 0.0:
                    return True
            except ValueError:
                pass
        total_match = _LLAMACPP_DOWNLOAD_TOTAL_RE.search(text)
        if total_match:
            try:
                if float(total_match.group(1)) > 0.0:
                    return True
            except ValueError:
                pass
        pct = DeployLogSummarizer._llamacpp_progress_percent(text)
        return pct is not None and pct > 0

    @staticmethod
    def _llamacpp_progress_percent(text: str) -> int | None:
        pct_match = _LLAMACPP_DOWNLOAD_PCT_RE.search(text)
        if pct_match is None:
            return None
        try:
            pct = int(pct_match.group(1))
        except ValueError:
            return None
        return max(0, min(100, pct))

    @staticmethod
    def _hf_fetch_percent(text: str) -> int | None:
        match = _HF_FETCH_PCT_RE.search(text)
        if match is None:
            return None
        try:
            return max(0, min(100, int(match.group(1))))
        except ValueError:
            return None
