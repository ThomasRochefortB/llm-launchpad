"""Runtime construction shared by providers that bootstrap their own hosts.

Modal deploys an app whose entrypoint owns these decisions. Prime and Vast
hand a container a command line, so both have to build the same one. Anything
here has two real callers; provider-specific bootstrap stays with its provider.
"""

from __future__ import annotations

import hashlib
import shlex

from ..protocol.models import DeploymentConfig
from .hf_download import hf_download_env


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


def gguf_stage_command(
    *,
    repo_id: str,
    revision: str | None,
    quant: str,
    dest_dir: str,
    cache_dir: str | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Build the shell prelude that stages exact GGUF weights with hf-xet.

    Returns the setup fragment and the repeated ``--hf-file`` flags llama.cpp
    should open. The pinned ``huggingface_hub`` inside the runtime images
    already depends on ``hf-xet``; when the interpreter lacks it, the script
    installs the accelerator before downloading.

    llama.cpp's ``--hf-repo`` reimplements the download over plain HTTP with
    one task per file, so staging through ``huggingface_hub`` first is the
    accelerated replacement. It resolves every shard of the matched quant
    (mirroring Modal's own allow-pattern selection), then passes each cached
    file back to llama.cpp via ``--hf-file``. The file still has to exist in
    the repo listing for llama.cpp to accept it; staging guarantees that
    because both sides list the same Hub repo.
    """
    repo = repo_id.strip()
    quant_text = quant.strip()
    if not repo or "/" not in repo:
        raise ValueError("Staged GGUF downloads require a 'owner/repo' id.")
    if not quant_text:
        raise ValueError("Staged GGUF downloads require a quant name.")
    dest = dest_dir.rstrip("/") or dest_dir
    cache = (cache_dir or f"{dest}/hf-hub").rstrip("/") or dest
    quoted_dest = shlex.quote(dest)
    quoted_cache = shlex.quote(cache)
    setup = (
        f"mkdir -p {quoted_dest} {quoted_cache} || exit $?; "
        f"export LLM_LAUNCHPAD_GGUF_REPO={shlex.quote(repo)}; "
        f"export LLM_LAUNCHPAD_GGUF_QUANT={shlex.quote(quant_text)}; "
        f"export LLM_LAUNCHPAD_GGUF_DEST={quoted_dest}; "
        f"export LLM_LAUNCHPAD_GGUF_CACHE={quoted_cache}; "
        + (
            f"export LLM_LAUNCHPAD_GGUF_REVISION={shlex.quote(str(revision).strip())}; "
            if (revision or "").strip()
            else "export LLM_LAUNCHPAD_GGUF_REVISION=; "
        )
        + "export HF_HUB_CACHE=\"$LLM_LAUNCHPAD_GGUF_CACHE\"; "
        + " ".join(
            f"export {name}={shlex.quote(value)}"
            for name, value in sorted(hf_download_env().items())
            if name.startswith(("HF_HUB_", "HF_XET_"))
        )
        + " "
        # Shares one downloader across two shells (Prime runs sh, Vast runs
        # it too) without heredoc quoting fights: python reads argv, never
        # stdin, so neither shell interprets the program text.
        + "python3 - \"$LLM_LAUNCHPAD_GGUF_REPO\" \"$LLM_LAUNCHPAD_GGUF_QUANT\" "
        "\"$LLM_LAUNCHPAD_GGUF_DEST\" \"$LLM_LAUNCHPAD_GGUF_CACHE\" "
        "\"$LLM_LAUNCHPAD_GGUF_REVISION\" "
        + shlex.quote(_GGUF_STAGE_PROGRAM)
        + " || exit $?; "
    )
    return setup, ("--hf-file", f"{dest}/weights.args")


_GGUF_STAGE_PROGRAM = r"""
import sys
from pathlib import Path

repo_id, quant, dest_dir, cache_dir, revision = sys.argv[1:6]
revision = revision or None
dest = Path(dest_dir)
dest.mkdir(parents=True, exist_ok=True)

try:
    import hf_xet  # noqa: F401
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "hf-xet"], check=True)

from huggingface_hub import HfApi, snapshot_download

info = HfApi().model_info(repo_id, revision=revision, files_metadata=True, timeout=30)
folded = quant.casefold()
names = [
    s.rfilename for s in (info.siblings or [])
    if s.rfilename.lower().endswith(".gguf")
    and folded in s.rfilename.casefold()
    and all(m not in s.rfilename.casefold() for m in ("mmproj", "imatrix", "draft", "eagle"))
]
if not names:
    raise SystemExit(f"No GGUF files match quant {quant!r} in {repo_id!r}.")
patterns = sorted({f"*{n.rsplit('/', 1)[-1]}*" for n in names})
snapshot = Path(snapshot_download(
    repo_id=repo_id, revision=revision or info.sha,
    cache_dir=cache_dir, allow_patterns=patterns,
))
files = sorted(p for n in names for p in [snapshot / n] if p.is_file())
missing = sorted(set(names) - {str(p.relative_to(snapshot)) for p in files})
if missing:
    raise SystemExit(f"Downloaded snapshot is missing {len(missing)} file(s): {missing[0]} ...")
# llama-server accepts one --hf-file plus sibling shards resolved from the
# same repo listing, so emit the first shard's repo-relative path and keep the
# full manifest alongside for operators. Every shard is already on disk.
first = sorted(files)[0].relative_to(snapshot).as_posix()
(dest / "weights.args").write_text(first + "\n", encoding="utf-8")
(dest / "weights.list").write_text("\n".join(str(p) for p in files) + "\n", encoding="utf-8")
print(f"staged {len(files)} GGUF file(s) from {repo_id}")
""".strip()


def gguf_stage_args(
    *,
    repo_id: str,
    revision: str | None,
    quant: str,
    dest_dir: str,
    cache_dir: str | None = None,
    extra_args: tuple[str, ...] = (),
) -> tuple[list[str], str]:
    """Stage weights, then serve them via ``--hf-repo`` + ``--hf-file``.

    Returns the server argv prefix (ending before caller flags) and the setup
    prelude that must run first. The repo flag keeps llama.cpp's listing,
    cache layout, and sibling-shard resolution intact; the file flag pins the
    exact quant instead of its quant-guessing fallback. Split shards stay
    complete because staging resolves every shard of the matched quant first.
    """
    setup, flag = gguf_stage_command(
        repo_id=repo_id, revision=revision, quant=quant,
        dest_dir=dest_dir, cache_dir=cache_dir,
    )
    args = ["/app/llama-server", "--hf-repo", repo_id.strip(), flag[0], flag[1], *extra_args]
    return args, setup


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
