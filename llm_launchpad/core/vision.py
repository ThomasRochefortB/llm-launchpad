"""Revision-specific image capability discovery and deployment validation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import PurePosixPath
import shlex
from typing import Any

from ..protocol.enums import BackendType, VisionMode, VisionVerification
from ..protocol.models import DeploymentConfig, ProjectorArtifact, VisionCapabilities
from .coerce import positive_int
from .hf_models import _load_repo_json_file


def vision_to_dict(vision: VisionCapabilities | None) -> dict[str, Any] | None:
    """Serialize effective image support, including its verification identity."""
    return asdict(vision) if vision is not None else None


def vision_from_dict(raw: Any) -> VisionCapabilities | None:
    """Read optional image metadata; old or corrupt records remain unverified."""
    if not isinstance(raw, dict):
        return None
    try:
        artifact = raw.get("projector")
        projector = None
        if isinstance(artifact, dict):
            projector = ProjectorArtifact(
                repo_id=str(artifact["repo_id"]), revision=str(artifact["revision"]),
                filename=str(artifact["filename"]), size_bytes=positive_int(artifact.get("size_bytes")),
            )
        supported = raw.get("supported")
        result = VisionCapabilities(
            supported=supported if isinstance(supported, bool) else None,
            enabled=raw.get("enabled") is True,
            model_revision=raw.get("model_revision"), runtime_id=raw.get("runtime_id"),
            projector=projector, message=str(raw.get("message") or "Image capability unknown."),
            fingerprint=str(raw.get("fingerprint") or ""),
            verification=VisionVerification(raw.get("verification", "untested")),
        )
        if not result.fingerprint or not result.enabled:
            result.verification = VisionVerification.UNTESTED
        return result
    except (KeyError, TypeError, ValueError):
        return None


def image_input_verified(vision: VisionCapabilities | None) -> bool:
    """Only advertise image input after a request reached the deployed model."""
    return bool(vision and vision.enabled and vision.fingerprint
                and vision.verification == VisionVerification.PASSED)


def _siblings(info: Any) -> dict[str, int | None]:
    return {str(row.rfilename): positive_int(getattr(row, "size", None))
            for row in (getattr(info, "siblings", None) or [])}


def select_projector(files: dict[str, int | None], filename: str | None = None) -> str:
    """Select a single projector without guessing among incompatible variants."""
    if filename:
        path = PurePosixPath(filename)
        if path.is_absolute() or ".." in path.parts or filename not in files:
            raise ValueError("Projector filename must identify a file in the selected HF revision.")
        if not filename.lower().endswith(".gguf"):
            raise ValueError("The projector must be a GGUF file.")
        return filename
    candidates = sorted(name for name in files
                        if "mmproj" in name.lower() and name.lower().endswith(".gguf"))
    if len(candidates) != 1:
        detail = ", ".join(candidates) or "none found"
        raise ValueError(f"Select --projector-file (and --projector-repo if needed): {detail}.")
    return candidates[0]


def inspect_model_vision(repo_id: str, revision: str | None = None) -> tuple[VisionCapabilities, dict[str, int | None]]:
    """Inspect pinned repository metadata without importing remote model code."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True, timeout=10)
    sha = str(info.sha or "").strip()
    if not sha:
        raise ValueError("Hugging Face did not resolve a model revision.")
    files = _siblings(info)
    config = _load_repo_json_file(repo_id, sha, "config.json") if "config.json" in files else None
    processor = (_load_repo_json_file(repo_id, sha, "preprocessor_config.json")
                 if "preprocessor_config.json" in files else None)
    has_projector = any("mmproj" in name.lower() and name.lower().endswith(".gguf") for name in files)
    has_encoder = isinstance(config, dict) and any(
        isinstance(config.get(key), dict) and bool(config[key])
        for key in ("vision_config", "visual", "visual_config")
    )
    has_processor = isinstance(processor, dict) and bool(processor.get("image_processor_type"))
    supported = True if has_projector or has_encoder or has_processor else None
    # A complete plain decoder configuration with no image processor is negative evidence.
    if supported is None and config and config.get("architectures") and not has_processor:
        supported = False if config.get("model_type") in {
            "llama", "qwen2", "qwen3", "qwen3_moe", "mistral", "deepseek_v2", "deepseek_v3",
        } else None
    return VisionCapabilities(
        supported=supported, model_revision=sha,
        message="Image encoder/projector detected." if supported else "Image capability is unknown." if supported is None else "Text-only model configuration.",
    ), files


def validate_vision_options(config: DeploymentConfig) -> None:
    """Reject invalid or conflicting inputs before allocating compute."""
    config.vision_mode = VisionMode(config.vision_mode)
    if config.image_limit < 1:
        raise ValueError("Image limit must be at least one; use --vision off for text-only mode.")
    if config.mm_processor_kwargs:
        try:
            value = json.loads(config.mm_processor_kwargs)
        except ValueError as exc:
            raise ValueError("Multimodal processor kwargs must be a JSON object.") from exc
        if not isinstance(value, dict):
            raise ValueError("Multimodal processor kwargs must be a JSON object.")
    projector_flags = {"--mmproj", "--mmproj-url", "--mmproj-auto", "--no-mmproj", "--no-mmproj-auto", "-mm", "-mmu", "--no-mmproj-offload"}
    if any(arg.split("=", 1)[0] in projector_flags for arg in shlex.split(config.server_args or "")):
        raise ValueError("Use --vision and --projector-* controls instead of raw projector arguments.")
    if config.backend == BackendType.VLLM and any((config.projector_repo, config.projector_file, config.projector_revision)):
        raise ValueError("Projector overrides apply only to llama.cpp.")
    if config.backend == BackendType.LLAMACPP and (config.mm_processor_kwargs or config.image_limit != 1):
        raise ValueError("Image limits and processor kwargs apply only to vLLM.")
    if config.vision_mode == VisionMode.OFF and any((config.projector_repo, config.projector_file, config.projector_revision)):
        raise ValueError("Projector overrides conflict with --vision off.")


def prepare_vision(config: DeploymentConfig) -> VisionCapabilities:
    """Resolve effective image input and immutable projector identity before deploy."""
    validate_vision_options(config)
    repo = config.repo_id if config.backend == BackendType.LLAMACPP else config.model_name
    revision = config.revision if config.backend == BackendType.LLAMACPP else config.model_revision
    if not repo and config.preset:
        from ..presets import PRESETS
        preset = PRESETS.get(config.preset, {})
        repo, revision = preset.get("repo_id"), preset.get("revision")
    vision = VisionCapabilities()
    files: dict[str, int | None] = {}
    if config.vision_mode != VisionMode.OFF and repo:
        try:
            vision, files = inspect_model_vision(repo, revision)
        except Exception as exc:
            vision.message = f"Image capability inspection unavailable: {exc}"
    explicit_projector = bool(config.projector_repo or config.projector_file or config.projector_revision)
    vision.enabled = config.vision_mode == VisionMode.ON or (
        config.vision_mode == VisionMode.AUTO and (vision.supported is True or explicit_projector)
    )
    if vision.enabled and vision.supported is False and not explicit_projector:
        raise ValueError("The selected model is text-only; image input cannot be enabled.")
    if vision.enabled and config.backend == BackendType.LLAMACPP:
        from huggingface_hub import HfApi
        projector_repo = config.projector_repo or repo
        if not projector_repo:
            raise ValueError("Vision requires a model repository and a projector.")
        projector_revision = config.projector_revision or (vision.model_revision if projector_repo == repo else None)
        if projector_repo != repo or config.projector_revision or not vision.model_revision:
            info = HfApi().model_info(projector_repo, revision=projector_revision, files_metadata=True, timeout=10)
            projector_revision, files = info.sha, _siblings(info)
        if not projector_revision:
            raise ValueError("Could not pin the projector revision.")
        filename = select_projector(files, config.projector_file)
        vision.projector = ProjectorArtifact(projector_repo, projector_revision, filename, files[filename])
    if config.vision_mode == VisionMode.OFF:
        vision.message = "Image input disabled (text-only deployment)."
    vision.runtime_id = config.llamacpp_runtime_id if config.backend == BackendType.LLAMACPP else "vllm-0.19.1"
    identity = {
        "repo": repo, "revision": vision.model_revision or revision, "quant": config.quant,
        "runtime": vision.runtime_id, "mode": config.vision_mode.value,
        "projector": asdict(vision.projector) if vision.projector else None,
        "image_limit": config.image_limit, "processor": config.mm_processor_kwargs,
        "args": config.server_args, "provider": config.provider.value,
    }
    vision.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    config.vision = vision
    return vision


def vllm_vision_limits(config: DeploymentConfig) -> str:
    """Return native vLLM modality limits for this image-only integration."""
    enabled = config.vision.enabled if config.vision else config.vision_mode != VisionMode.OFF
    return json.dumps({"image": config.image_limit if enabled else 0, "video": 0, "audio": 0})
