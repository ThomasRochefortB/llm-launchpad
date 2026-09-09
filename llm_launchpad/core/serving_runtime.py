"""Runtime construction shared by providers that bootstrap their own hosts.

Modal deploys an app whose entrypoint owns these decisions. Prime and Vast
hand a container a command line, so both have to build the same one. Anything
here has two real callers; provider-specific bootstrap stays with its provider.
"""

from __future__ import annotations

import hashlib
import shlex

from ..protocol.models import DeploymentConfig


def vllm_serve_args(config: DeploymentConfig, *, host: str, port: int) -> tuple[str, ...]:
    """Build the `vllm serve` command line for one deployment.

    The API key is deliberately absent: vLLM reads ``VLLM_API_KEY`` from the
    environment, and a key in argv is readable by anything that can list
    processes on the host. Callers that need it on the command line append it
    themselves.
    """

    from .vision import vllm_vision_limits

    model_name = str(config.model_name or "").strip()
    served_name = str(config.served_model_name or model_name.rsplit("/", 1)[-1])
    args = [
        "vllm", "serve", model_name,
        "--host", host,
        "--port", str(port),
        "--uvicorn-log-level", "info",
        "--served-model-name", served_name,
        "--tensor-parallel-size", str(config.n_gpu or config.gpu_count or 1),
        "--limit-mm-per-prompt", vllm_vision_limits(config),
    ]
    if config.mm_processor_kwargs:
        args.extend(["--mm-processor-kwargs", config.mm_processor_kwargs])
    if config.model_revision:
        args.extend(["--revision", config.model_revision])
    if config.trust_remote_code:
        args.append("--trust-remote-code")
    if config.fast_boot:
        args.append("--enforce-eager")
    if config.reasoning_parser:
        args.extend(["--reasoning-parser", config.reasoning_parser])
    if config.tool_call_parser:
        args.extend(["--enable-auto-tool-choice", "--tool-call-parser", config.tool_call_parser])
    if config.default_chat_template_kwargs:
        args.extend(["--default-chat-template-kwargs", config.default_chat_template_kwargs])
    return tuple(args)


def projector_setup(config: DeploymentConfig, *, root: str) -> tuple[str, str]:
    """Stage the exact projector atomically under ``root``.

    Returns the shell prelude and the resulting path. Both llama.cpp runtimes
    ship curl, and ``$HF_TOKEN`` survives shlex.join literally so the container
    shell expands it; the header is omitted entirely when no token is set,
    because an empty Bearer makes Hugging Face reject public files that an
    unauthenticated request would have served.
    """

    from huggingface_hub import hf_hub_url

    artifact = config.vision.projector if config.vision else None
    if artifact is None:
        raise ValueError("Vision enabled without a resolved projector.")
    identity = hashlib.sha256(
        f"{artifact.repo_id}@{artifact.revision}/{artifact.filename}".encode()
    ).hexdigest()
    path = f"{root}/projectors/{identity}.gguf"
    quoted = shlex.quote(path)
    url = hf_hub_url(artifact.repo_id, artifact.filename, revision=artifact.revision)
    size_check = (
        f'[ "$(stat -c %s {quoted})" = {artifact.size_bytes} ]'
        if artifact.size_bytes
        else f"[ -s {quoted} ]"
    )
    setup = (
        f"mkdir -p {shlex.quote(root + '/projectors')} || exit $?; "
        f"if ! {size_check}; then "
        f"curl --fail --location --retry 3 --connect-timeout 20 "
        '${HF_TOKEN:+--header "Authorization: Bearer $HF_TOKEN"} '
        f'{shlex.quote(url)} -o {quoted}.tmp '
        f"&& mv {quoted}.tmp {quoted} || exit $?; fi; "
        f"{size_check} || exit 1; "
        f'[ "$(head -c 4 {quoted})" = GGUF ] || exit 1; '
    )
    return setup, path
