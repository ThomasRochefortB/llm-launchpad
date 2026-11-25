"""Portable Prime runtime command builders, tunnel scripts, and log/progress parsers."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
import re
import shlex
from typing import Any

from ..protocol.enums import BackendType
from ..protocol.models import DeploymentConfig
from .deploy_log_summary import is_error_like
from .prime_offers import PrimeLaunchSpec
from .provider_options import prime_provider_options
from .runtime_support import llamacpp_cuda_architecture
from .serving_runtime import engram_setup, projector_setup, vllm_serve_args

PRIME_RUNTIME_CONTAINER_NAME = "llm-launchpad-runtime"
PRIME_RUNTIME_ROOT = "/opt/llm-launchpad"
PRIME_TUNNEL_LABEL = "llm-launchpad"
PRIME_TUNNEL_PID_PATH = f"{PRIME_RUNTIME_ROOT}/tunnel.pid"
PRIME_TUNNEL_ID_NAME = "tunnel-id"
PRIME_TUNNEL_LOG_PATH = f"{PRIME_RUNTIME_ROOT}/tunnel.log"

LLAMACPP_FIT_LOG_VERBOSITY = 4
PRIME_FIT_FILE_VARIABLE = "LLAMA_ARG_FIT_FILE"

# cmake prints "[ 45%] Building CXX object ..." as it compiles; docker build
# prefixes each line with its step number, so the marker is matched anywhere.
_CMAKE_PERCENT_RE = re.compile(r"\[\s*(\d{1,3})%\]")
# llama.cpp prefixes its own warnings with a W level and a "warn:" note, and
# both carry words an error match would accept.
_RUNTIME_WARNING_RE = re.compile(r"^\s*(warn(ing)?[: ]|\S+\s+W\s)", re.IGNORECASE)


def _gigabytes(byte_count: int) -> str:
    """Format a byte count in decimal GB, the unit model weights are quoted in."""
    return f"{byte_count / 1000**3:.1f}"


def _prime_projector_setup(config: DeploymentConfig) -> tuple[str, str]:
    """Stage the exact projector atomically in the persistent llama cache."""
    root = "/data/llama.cpp" if prime_provider_options(config).disk_id else "/root/.cache/llama.cpp"
    return projector_setup(config, root=root)


def _prime_engram_setup(config: DeploymentConfig) -> tuple[str, str]:
    """Stage the exact external Engram (--ple) table in the persistent llama cache."""
    root = "/data/llama.cpp" if prime_provider_options(config).disk_id else "/root/.cache/llama.cpp"
    return engram_setup(config, root=root)


def llamacpp_fit_relevant_args(arguments: list[str]) -> list[str]:
    """Keep memory-affecting common args accepted by llama-fit-params."""

    supported = {
        "-c",
        "--ctx-size",
        "-b",
        "--batch-size",
        "-ub",
        "--ubatch-size",
        "-ctk",
        "--cache-type-k",
        "-ctv",
        "--cache-type-v",
        "-ngl",
        "--n-gpu-layers",
        "--gpu-layers",
        "-fa",
        "--flash-attn",
        "-fitt",
        "--fit-target",
    }
    result: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token.split("=", 1)[0] in supported:
            result.append(token)
            if "=" not in token and index + 1 < len(arguments):
                result.append(arguments[index + 1])
                index += 2
                continue
        index += 1
    return result


def runtime_failure_detail(logs: Sequence[str], fallback: str) -> str:
    """Pick the line that explains a dead runtime, not the one printed last."""
    candidates = [line.strip() for line in logs if line and line.strip()]
    if not candidates:
        return fallback
    for line in candidates:
        if _RUNTIME_WARNING_RE.search(line):
            continue
        if is_error_like(line):
            return line
    return candidates[-1]


def build_runtime_env(config: DeploymentConfig) -> dict[str, str]:
    """Build container environment variables for a Prime vLLM or llama.cpp launch."""
    from .hf_download import hf_download_env

    options = prime_provider_options(config)
    if config.backend == BackendType.VLLM:
        env: dict[str, str] = {
            "MODEL_NAME": (config.model_name or "").strip(),
            "SERVED_MODEL_NAME": (config.served_model_name or "").strip(),
            "N_GPU": str(config.n_gpu or config.gpu_count or 1),
            "TRUST_REMOTE_CODE": "true" if config.trust_remote_code else "false",
            "FAST_BOOT": "true" if config.fast_boot else "false",
            "VLLM_API_KEY": (config.endpoint_api_key or "").strip(),
        }
        optional = {
            "MODEL_REVISION": config.model_revision,
            "REASONING_PARSER": config.reasoning_parser,
            "TOOL_CALL_PARSER": config.tool_call_parser,
            "DEFAULT_CHAT_TEMPLATE_KWARGS": config.default_chat_template_kwargs,
        }
        env.update(
            {
                key: value.strip()
                for key, value in optional.items()
                if value and value.strip()
            }
        )
        env.update(
            {
                key: value
                for key, value in hf_download_env().items()
                if key.startswith(("HF_HUB_", "HF_XET_"))
            }
        )
        if options.disk_id:
            env["HF_HOME"] = "/data/huggingface"
            env["VLLM_CACHE_ROOT"] = "/data/vllm"
    else:
        extra_args = shlex.split(config.server_args or "")
        env = {
            "REPO_ID": (config.repo_id or "").strip(),
            "QUANT": (config.quant or "").strip(),
            "SERVED_MODEL_NAME": (config.served_model_name or "").strip(),
            "LLAMACPP_API_KEY": (config.endpoint_api_key or "").strip(),
            "LLAMACPP_SERVER_ARGS": "\n".join(extra_args),
        }
        if config.placement_assessment is not None:
            env["LLM_LAUNCHPAD_SERVING_FINGERPRINT"] = (
                config.placement_assessment.fingerprint
            )
        if config.n_gpu_layers is not None:
            env["N_GPU_LAYERS"] = str(config.n_gpu_layers)
        if options.disk_id:
            env["LLAMA_CACHE"] = "/data/llama.cpp"
    try:
        from huggingface_hub import get_token

        hf_token = (get_token() or "").strip()
    except Exception:
        hf_token = ""
    if hf_token:
        env["HF_TOKEN"] = hf_token
    return {key: value for key, value in env.items() if value}


def build_docker_env_file(config: DeploymentConfig, runtime_env: dict[str, str]) -> str:
    """Format the ``--env-file`` content staged on the Prime pod."""
    values: dict[str, str] = {}
    endpoint_api_key = str(config.endpoint_api_key or "").strip()
    if not endpoint_api_key:
        raise ValueError("Prime portable runtimes require an endpoint API key.")
    if runtime_env.get("HF_TOKEN"):
        values["HF_TOKEN"] = runtime_env["HF_TOKEN"]
    if config.backend == BackendType.LLAMACPP:
        values["LLAMA_ARG_API_KEY"] = endpoint_api_key
        fingerprint = str(
            runtime_env.get("LLM_LAUNCHPAD_SERVING_FINGERPRINT") or ""
        ).strip()
        if fingerprint:
            values["LLM_LAUNCHPAD_SERVING_FINGERPRINT"] = fingerprint
        if prime_provider_options(config).disk_id:
            values["LLAMA_CACHE"] = "/data/llama.cpp"
    else:
        values["VLLM_API_KEY"] = endpoint_api_key
        for key in (
            "HF_HUB_ETAG_TIMEOUT",
            "HF_HUB_DOWNLOAD_TIMEOUT",
            "HF_HUB_DISABLE_XET",
            "HF_XET_HIGH_PERFORMANCE",
        ):
            if runtime_env.get(key):
                values[key] = runtime_env[key]
        if prime_provider_options(config).disk_id:
            values["HF_HOME"] = "/data/huggingface"
            values["VLLM_CACHE_ROOT"] = "/data/vllm"
    for key, value in values.items():
        if "\n" in value or "\r" in value:
            raise ValueError(f"Prime runtime value {key} may not contain newlines.")
    return "".join(f"{key}={value}\n" for key, value in values.items() if value)


def build_probe_config(config: DeploymentConfig) -> str:
    """Build the curl config file used to probe localhost:8000 inside the pod."""
    api_key = str(config.endpoint_api_key or "")
    if any(character in api_key for character in {'"', "\\", "\n", "\r"}):
        raise ValueError(
            "Endpoint API keys may not contain quotes, backslashes, or newlines on Prime."
        )
    return (
        "silent\nshow-error\nfail\nmax-time = 5\n"
        f'header = "Authorization: Bearer {api_key}"\n'
    )


def _llamacpp_docker_command(
    command: list[str],
    config: DeploymentConfig,
    launch: PrimeLaunchSpec,
) -> list[str]:
    """Append llama.cpp staging, fit guard, and server entrypoint args to ``command``."""
    from .serving_runtime import (
        GGUF_FILE_PLACEHOLDER,
        PRIME_API_KEY_PLACEHOLDER,
        gguf_exec_command,
        gguf_stage_command,
    )

    options = prime_provider_options(config)
    repo = str(config.repo_id or "").strip()
    quant = str(config.quant or "").strip()
    if not repo or not quant:
        raise ValueError("Prime llama.cpp runtimes require a GGUF repository and quant.")
    hf_repo = f"{repo}:{quant}" if quant else repo
    stage_root = "/data/llama.cpp" if options.disk_id else "/root/.cache/llama.cpp"
    excluded_shards = (
        [config.engram_table.replaced_shard]
        if config.engram_table is not None and config.engram_table.replaced_shard
        else None
    )
    stage_setup, hf_file_flag = gguf_stage_command(
        repo_id=repo,
        revision=None,
        quant=quant,
        dest_dir=f"{stage_root}/staged",
        cache_dir=f"{stage_root}/staged/hf-hub",
        exclude_files=excluded_shards,
    )
    llama_args = [
        "/app/llama-server",
        "--hf-repo",
        hf_repo,
        hf_file_flag[0],
        GGUF_FILE_PLACEHOLDER,
    ]
    if config.n_gpu_layers is not None:
        llama_args.extend(["--n-gpu-layers", str(config.n_gpu_layers)])
    server_extra_args = shlex.split(config.server_args or "")
    llama_args.extend(
        [
            *server_extra_args,
            "--metrics",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--alias",
            str(config.served_model_name or repo.rsplit("/", 1)[-1]),
        ]
    )
    fit_args = [
        "/app/llama-fit-params",
        "--hf-repo",
        hf_repo,
        hf_file_flag[0],
        GGUF_FILE_PLACEHOLDER,
        "--verbosity",
        str(LLAMACPP_FIT_LOG_VERBOSITY),
        *llamacpp_fit_relevant_args(server_extra_args),
    ]
    projector_setup = stage_setup
    if config.engram_table is not None:
        engram_prelude, engram_path = _prime_engram_setup(config)
        projector_setup += engram_prelude
        llama_args.extend(["--ple", engram_path])
        fit_args.extend(["--ple", engram_path])
    if config.vision is not None:
        if config.vision.enabled:
            vision_setup, projector_path = _prime_projector_setup(config)
            projector_setup += vision_setup
            llama_args.extend(["--mmproj", projector_path])
        else:
            llama_args.append("--no-mmproj")
    attestation_marker = (
        "echo LLM_LAUNCHPAD_ATTESTATION_JSON_BEGIN; "
        "printf '{\"schema_version\":1,\"fingerprint\":\"%s\","
        "\"gpu_only_requested\":true}\\n' "
        '"$LLM_LAUNCHPAD_SERVING_FINGERPRINT"; '
        "echo LLM_LAUNCHPAD_ATTESTATION_JSON_END; "
    )
    fit_guard = ""
    if config.placement_assessment is not None:
        joined_fit = shlex.join(fit_args).replace(
            shlex.quote(GGUF_FILE_PLACEHOLDER),
            f'"${PRIME_FIT_FILE_VARIABLE}"',
        )
        fit_guard = (
            f"{PRIME_FIT_FILE_VARIABLE}="
            f"$(cat {shlex.quote(hf_file_flag[1])} 2>/dev/null || true); "
            f"if [ -x /app/llama-fit-params ] && "
            f"[ -n \"${PRIME_FIT_FILE_VARIABLE}\" ]; then {joined_fit} "
            "|| exit $?; fi; "
            f"{attestation_marker}"
        )
    inner = projector_setup + fit_guard + gguf_exec_command(
        [*llama_args, "--api-key", PRIME_API_KEY_PLACEHOLDER],
        weights_args_path=hf_file_flag[1],
    ).replace(
        shlex.quote(PRIME_API_KEY_PLACEHOLDER), '"$LLAMA_ARG_API_KEY"'
    )
    if not options.disk_id:
        command.extend(["-v", "llm-launchpad-llama-cache:/root/.cache/llama.cpp"])
    command.extend(
        ["--env", "LLAMA_CACHE=/root/.cache/llama.cpp/staged/hf-hub"]
        if not options.disk_id
        else ["--env", "LLAMA_CACHE=/data/llama.cpp/staged/hf-hub"]
    )
    command.extend(["--entrypoint", "/bin/sh", launch.container_image, "-lc", inner])
    return command


def build_bootstrap_docker_command(
    config: DeploymentConfig,
    launch: PrimeLaunchSpec,
) -> list[str]:
    """Build the ``docker run`` invocation for Prime's portable runtime."""
    options = prime_provider_options(config)
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        PRIME_RUNTIME_CONTAINER_NAME,
        "--restart",
        "unless-stopped",
        "--gpus",
        "all",
        "--ipc=host",
        "-p",
        "8000:8000" if options.allow_insecure_http else "127.0.0.1:8000:8000",
        "--env-file",
        f"{PRIME_RUNTIME_ROOT}/runtime.env",
        "-v",
        "llm-launchpad-hf-cache:/root/.cache/huggingface",
    ]
    if options.disk_id:
        command.extend(["-v", "/data:/data"])

    if config.backend == BackendType.LLAMACPP:
        return _llamacpp_docker_command(command, config, launch)

    vllm_args = list(vllm_serve_args(config, host="0.0.0.0", port=8000))
    inner = f'exec {shlex.join(vllm_args)} --api-key "$VLLM_API_KEY"'
    command.extend(
        [
            "--entrypoint",
            "/bin/bash",
            launch.container_image,
            "-lc",
            inner,
        ]
    )
    return command


def build_bootstrap_script(
    config: DeploymentConfig,
    launch: PrimeLaunchSpec,
    docker_command: list[str],
) -> str:
    """Build the pod-side bootstrap shell script that prepares and starts Docker."""
    prepare_image = (
        f"if docker image inspect {shlex.quote(launch.container_image)} "
        f">/dev/null 2>&1; then echo 'Serving image already present; "
        f"skipping pull'; else docker pull {shlex.quote(launch.container_image)}; fi"
    )
    if launch.build_recipe:
        jobs = os.cpu_count() or 4
        cuda_arch = llamacpp_cuda_architecture(config.gpu_type)
        build_flags = f"--build-arg CUDA_ARCHITECTURES={cuda_arch} " if cuda_arch else ""
        build_flags += f"--build-arg BUILD_JOBS={jobs} "
        prepare_image = (
            f"docker build -t {shlex.quote(launch.container_image)} "
            f"{build_flags}- < {PRIME_RUNTIME_ROOT}/runtime.dockerfile"
        )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"rm -f {PRIME_RUNTIME_ROOT}/bootstrap.exit",
            (
                "trap 'status=$?; printf \"%s\\n\" \"$status\" > "
                f"{PRIME_RUNTIME_ROOT}/bootstrap.exit' EXIT"
            ),
            (
                "for attempt in $(seq 1 36); do "
                "if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; "
                "then break; fi; "
                'echo "Waiting for the Prime image to finish installing Docker '
                '($attempt/36)"; sleep 5; done'
            ),
            (
                "command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 || "
                "{ echo 'Prime image did not provide a ready Docker runtime after 180 seconds' "
                ">&2; exit 127; }"
            ),
            "nvidia-smi >/dev/null",
            prepare_image,
            f"docker rm -f {PRIME_RUNTIME_CONTAINER_NAME} >/dev/null 2>&1 || true",
            shlex.join(docker_command),
            "",
        ]
    )


def build_tunnel_config(tunnel: Any, *, local_port: int = 8000) -> str:
    """Build the minimal frpc configuration documented by Prime Tunnel."""

    if not 1 <= local_port <= 65535:
        raise ValueError("Prime Tunnel local port must be between 1 and 65535.")
    if not 1 <= tunnel.server_port <= 65535:
        raise ValueError("Prime Tunnel returned an invalid server port.")

    def toml_string(value: str) -> str:
        return json.dumps(value, ensure_ascii=True)

    return "\n".join(
        [
            f"serverAddr = {toml_string(tunnel.server_host)}",
            f"serverPort = {tunnel.server_port}",
            f"user = {toml_string(tunnel.tunnel_id)}",
            'auth.method = "token"',
            f"auth.token = {toml_string(tunnel.frp_token)}",
            f"metadatas.binding_secret = {toml_string(tunnel.binding_secret)}",
            "transport.tcpMux = true",
            "transport.tcpMuxKeepaliveInterval = 30",
            "transport.poolCount = 10",
            "transport.dialServerKeepalive = 60",
            'log.to = "console"',
            'log.level = "info"',
            "[[proxies]]",
            f"name = {toml_string(tunnel.tunnel_id)}",
            'type = "http"',
            'localIP = "127.0.0.1"',
            f"localPort = {local_port}",
            f"subdomain = {toml_string(tunnel.tunnel_id)}",
            "",
        ]
    )


def build_tunnel_start_script(*, cache_filename: str = "frpc") -> str:
    """Start a preinstalled frpc binary and cache it on an attached disk."""

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", cache_filename).strip("-") or "frpc"
    cached_target = f"/data/llm-launchpad/{safe_name}"
    cached_tmp = f"/data/llm-launchpad/.{safe_name}.tmp.$$"
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"if [ ! -x {PRIME_RUNTIME_ROOT}/frpc ]; then",
            '  echo "Prime Tunnel client is missing" >&2',
            "  exit 127",
            "fi",
            f"if [ -s {PRIME_TUNNEL_PID_PATH} ]; then",
            f'  old_pid="$(cat {PRIME_TUNNEL_PID_PATH})"',
            '  if kill -0 "$old_pid" 2>/dev/null; then kill "$old_pid"; fi',
            "fi",
            (
                f"nohup {PRIME_RUNTIME_ROOT}/frpc -c {PRIME_RUNTIME_ROOT}/tunnel.toml "
                f"> {PRIME_TUNNEL_LOG_PATH} 2>&1 </dev/null &"
            ),
            f'printf "%s\\n" "$!" > {PRIME_TUNNEL_PID_PATH}',
            "if [ -d /data ]; then",
            "  install -d -m 700 /data/llm-launchpad",
            f"  cp -f {PRIME_RUNTIME_ROOT}/frpc {cached_tmp} && mv -f {cached_tmp} {cached_target}",
            "fi",
            "",
        ]
    )


def redact_runtime_log(value: str) -> str:
    """Scrub API keys and tunnel secrets from Prime runtime/tunnel log lines."""
    value = re.sub(r"(--api-key(?:=|\s+))\S+", r"\1[redacted]", value)
    value = re.sub(
        (
            r"((?:HF_TOKEN|VLLM_API_KEY|LLAMA_ARG_API_KEY|"
            r"frp_token|binding_secret)=)\S+"
        ),
        r"\1[redacted]",
        value,
        flags=re.IGNORECASE,
    )
    return re.sub(
        (
            r"((?:['\"]?(?:api_key|token|frp_token|binding_secret)['\"]?)"
            r"\s*[=:]\s*)"
            r"(\[[^\]]*\]|'[^']*'|\"[^\"]*\"|\S+)"
        ),
        r"\1[redacted]",
        value,
        flags=re.IGNORECASE,
    )


def expected_model_bytes(config: DeploymentConfig) -> int:
    """Bytes the pod will actually download, for model-load percentages."""
    assessment = config.placement_assessment
    memory = getattr(assessment, "memory", None)
    weights_gb = getattr(memory, "weights_gb", None)
    if not weights_gb or weights_gb <= 0:
        return 0
    return int(float(weights_gb) * (1000**3))


def parse_docker_byte_count(value: str) -> int | None:
    """Parse docker stats sizes such as ``61.7GB`` or ``551MiB``."""
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE]i?B|B)\s*",
        value or "",
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        amount = float(match.group(1))
    except ValueError:
        return None
    unit = match.group(2).upper()
    multipliers = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "PB": 1000**5,
        "EB": 1000**6,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
        "PIB": 1024**5,
        "EIB": 1024**6,
    }
    multiplier = multipliers.get(unit)
    if multiplier is None:
        return None
    return int(amount * multiplier)


def parse_runtime_progress_output(stdout: str) -> tuple[str, int, int]:
    """Parse docker stats plus cache/expected byte counters."""
    network = ""
    cache_bytes = 0
    expected_bytes = 0
    for raw_line in (stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("CACHE:"):
            try:
                cache_bytes = max(0, int(float(line.split(":", 1)[1].strip() or "0")))
            except ValueError:
                cache_bytes = 0
            continue
        if line.startswith("EXPECTED:"):
            try:
                expected_bytes = max(0, int(float(line.split(":", 1)[1].strip() or "0")))
            except ValueError:
                expected_bytes = 0
            continue
        if not network:
            network = line
    return network, cache_bytes, expected_bytes


def format_runtime_load_detail(
    *,
    network: str,
    cache_bytes: int,
    expected_bytes: int,
    growing: bool | None = None,
) -> str:
    """Describe model download/load using cache bytes when a size is known."""
    downloaded = max(0, cache_bytes)
    rx_token = (network or "").split("/", 1)[0].strip()
    rx_bytes = parse_docker_byte_count(rx_token)
    if downloaded <= 0 and rx_bytes:
        downloaded = rx_bytes
    download_complete = expected_bytes > 0 and downloaded >= int(expected_bytes * 0.98)
    if downloaded > 0 and not download_complete:
        if expected_bytes > 0:
            percent = max(0, min(99, int(downloaded * 100 / expected_bytes)))
            return (
                f"runtime container is downloading the model ({percent}%), "
                f"{_gigabytes(downloaded)} of {_gigabytes(expected_bytes)} GB"
            )
        if growing:
            return (
                "runtime container is downloading the model, "
                f"{_gigabytes(downloaded)} GB so far"
            )
    if network.strip():
        return f"runtime container is loading the model (network {network.strip()})"
    return "runtime container is loading the model"


def parse_runtime_image_progress(stdout: str) -> tuple[str, bool, int | None]:
    """Return the bootstrap exit status, whether it builds, and its percent."""
    exit_status = ""
    building = False
    percent: int | None = None
    for raw_line in (stdout or "").splitlines():
        line = raw_line.strip()
        if line.startswith("EXIT:"):
            exit_status = line.split(":", 1)[1].strip()
            continue
        if line.startswith("KIND:"):
            building = line.split(":", 1)[1].strip() == "build"
            continue
        match = _CMAKE_PERCENT_RE.search(line)
        if match:
            try:
                percent = max(0, min(99, int(match.group(1))))
            except ValueError:
                percent = None
    return exit_status, building, percent
