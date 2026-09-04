"""Runtime compatibility checks for GGUF model recommendations and deploys."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from importlib import resources
import json
import re
from typing import Any
from collections.abc import Iterable

from .coerce import positive_int


DEFAULT_LLAMACPP_IMAGE_REF = "ghcr.io/ggml-org/llama.cpp:server-cuda-b10689"
LLAMACPP_SUPPORT_MANIFEST_FILENAME = "llamacpp_runtime_support.json"

_ARCHITECTURE_ROW_RE = re.compile(
    r'\{\s*(LLM_ARCH_[A-Z0-9_]+)\s*,\s*"([^"]+)"\s*\}'
)
_IGNORED_ARCHITECTURE_ENUMS = {"LLM_ARCH_CLIP", "LLM_ARCH_UNKNOWN"}
_MODEL_SYMBOL_RE = re.compile(
    r"\b(?:llm|llama)_(?:model|build)_([a-z0-9_]+)\b"
)


class RuntimeCompatibility(str, Enum):
    """Compatibility of one model architecture with a serving runtime."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LlamaCppSupportManifest:
    """Versioned architecture capabilities for one exact llama.cpp image."""

    runtime_id: str
    runtime_build: str
    image_ref: str
    image_digest: str
    source_revision: str
    source_url: str
    generated_at: str
    architectures: frozenset[str]
    compatible_image_refs: frozenset[str] = frozenset()
    mtp_architectures: frozenset[str] = frozenset()
    mtp_support_known: bool = False


@dataclass(frozen=True)
class RuntimeCompatibilityDecision:
    """Result of checking GGUF metadata against a runtime manifest."""

    status: RuntimeCompatibility
    architecture: str | None
    runtime_id: str
    runtime_build: str
    image_ref: str
    message: str

    @property
    def is_supported(self) -> bool:
        """Return whether the architecture can be recommended or deployed."""

        return self.status == RuntimeCompatibility.SUPPORTED


def extract_llamacpp_architectures(source: str) -> list[str]:
    """Extract loadable ``general.architecture`` values from llama-arch.cpp."""

    architectures = {
        architecture.strip().casefold()
        for enum_name, architecture in _ARCHITECTURE_ROW_RE.findall(source)
        if enum_name not in _IGNORED_ARCHITECTURE_ENUMS and architecture.strip()
    }
    return sorted(architectures)


def extract_llamacpp_mtp_architectures(
    architecture_source: str,
    model_sources: Iterable[str],
) -> list[str]:
    """Extract architectures whose pinned model code implements native MTP."""

    architecture_by_symbol = {
        re.sub(r"[^a-z0-9]+", "", enum_name.removeprefix("LLM_ARCH_").casefold()): architecture.strip().casefold()
        for enum_name, architecture in _ARCHITECTURE_ROW_RE.findall(architecture_source)
        if enum_name not in _IGNORED_ARCHITECTURE_ENUMS and architecture.strip()
    }
    supported: set[str] = set()
    for source in model_sources:
        if (
            "LLM_KV_NEXTN_PREDICT_LAYERS" not in source
            or "LLM_GRAPH_TYPE_DECODER_MTP" not in source
        ):
            continue
        for symbol in _MODEL_SYMBOL_RE.findall(source):
            architecture = architecture_by_symbol.get(
                re.sub(r"[^a-z0-9]+", "", symbol.casefold())
            )
            if architecture:
                supported.add(architecture)
    return sorted(supported)


@lru_cache(maxsize=1)
def load_llamacpp_support_manifest() -> LlamaCppSupportManifest:
    """Load and validate the bundled support manifest."""

    resource = resources.files("llm_launchpad.data").joinpath(
        LLAMACPP_SUPPORT_MANIFEST_FILENAME
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("llama.cpp support manifest must be a JSON object")

    schema_version = positive_int(payload.get("schema_version"))
    if schema_version not in {1, 2}:
        raise RuntimeError(
            f"Unsupported llama.cpp support manifest schema: {schema_version!r}"
        )

    runtime_id = _required_string(payload, "runtime_id")
    runtime_build = _required_string(payload, "runtime_build")
    image_ref = _required_string(payload, "image_ref")
    image_digest = _required_string(payload, "image_digest")
    source_revision = _required_string(payload, "source_revision")
    source_url = _required_string(payload, "source_url")
    generated_at = _required_string(payload, "generated_at")
    architectures = _string_set(payload.get("architectures"))
    if not architectures:
        raise RuntimeError("llama.cpp support manifest has no architectures")

    return LlamaCppSupportManifest(
        runtime_id=runtime_id,
        runtime_build=runtime_build,
        image_ref=image_ref,
        image_digest=image_digest,
        source_revision=source_revision,
        source_url=source_url,
        generated_at=generated_at,
        architectures=architectures,
        compatible_image_refs=_string_set(payload.get("compatible_image_refs")),
        mtp_architectures=_string_set(payload.get("mtp_architectures")),
        mtp_support_known=schema_version >= 2,
    )


def evaluate_llamacpp_architecture(
    architecture: str | None,
    *,
    image_ref: str | None = None,
    manifest: LlamaCppSupportManifest | None = None,
) -> RuntimeCompatibilityDecision:
    """Check one GGUF architecture against the exact configured llama.cpp image."""

    support = manifest or load_llamacpp_support_manifest()
    selected_image = (image_ref or support.image_ref).strip()
    normalized_architecture = _optional_string(architecture)
    if not _image_matches_manifest(selected_image, support):
        return RuntimeCompatibilityDecision(
            status=RuntimeCompatibility.UNKNOWN,
            architecture=normalized_architecture,
            runtime_id=support.runtime_id,
            runtime_build=support.runtime_build,
            image_ref=selected_image,
            message=(
                f"Compatibility is unknown for custom llama.cpp image "
                f"{selected_image!r}; the bundled support list targets "
                f"{support.image_ref!r}."
            ),
        )

    if normalized_architecture is None:
        return RuntimeCompatibilityDecision(
            status=RuntimeCompatibility.UNKNOWN,
            architecture=None,
            runtime_id=support.runtime_id,
            runtime_build=support.runtime_build,
            image_ref=selected_image,
            message=(
                "Hugging Face did not expose this GGUF's general.architecture, "
                "so llama.cpp compatibility could not be verified."
            ),
        )

    if normalized_architecture in support.architectures:
        return RuntimeCompatibilityDecision(
            status=RuntimeCompatibility.SUPPORTED,
            architecture=normalized_architecture,
            runtime_id=support.runtime_id,
            runtime_build=support.runtime_build,
            image_ref=selected_image,
            message=(
                f"GGUF architecture {normalized_architecture!r} is supported by "
                f"llama.cpp {support.runtime_build}."
            ),
        )

    return RuntimeCompatibilityDecision(
        status=RuntimeCompatibility.UNSUPPORTED,
        architecture=normalized_architecture,
        runtime_id=support.runtime_id,
        runtime_build=support.runtime_build,
        image_ref=selected_image,
        message=(
            f"GGUF architecture {normalized_architecture!r} is not recognized by "
            f"llama.cpp {support.runtime_build} ({support.image_ref})."
        ),
    )


def evaluate_llamacpp_mtp(
    architecture: str | None,
    nextn_predict_layers: int | None,
    *,
    image_ref: str | None = None,
    manifest: LlamaCppSupportManifest | None = None,
) -> RuntimeCompatibilityDecision:
    """Check embedded MTP metadata against one exact llama.cpp runtime."""

    support = manifest or load_llamacpp_support_manifest()
    selected_image = (image_ref or support.image_ref).strip()
    normalized_architecture = _optional_string(architecture)
    if not _image_matches_manifest(selected_image, support):
        return RuntimeCompatibilityDecision(
            status=RuntimeCompatibility.UNKNOWN,
            architecture=normalized_architecture,
            runtime_id=support.runtime_id,
            runtime_build=support.runtime_build,
            image_ref=selected_image,
            message=(
                f"MTP compatibility is unknown for custom llama.cpp image "
                f"{selected_image!r}."
            ),
        )
    if normalized_architecture is None or nextn_predict_layers is None:
        return RuntimeCompatibilityDecision(
            status=RuntimeCompatibility.UNKNOWN,
            architecture=normalized_architecture,
            runtime_id=support.runtime_id,
            runtime_build=support.runtime_build,
            image_ref=selected_image,
            message="Embedded MTP metadata could not be verified.",
        )
    if nextn_predict_layers <= 0:
        return RuntimeCompatibilityDecision(
            status=RuntimeCompatibility.UNSUPPORTED,
            architecture=normalized_architecture,
            runtime_id=support.runtime_id,
            runtime_build=support.runtime_build,
            image_ref=selected_image,
            message="The selected target GGUF has no embedded MTP heads.",
        )
    if not support.mtp_support_known:
        return RuntimeCompatibilityDecision(
            status=RuntimeCompatibility.UNKNOWN,
            architecture=normalized_architecture,
            runtime_id=support.runtime_id,
            runtime_build=support.runtime_build,
            image_ref=selected_image,
            message="The bundled runtime manifest predates MTP capability tracking.",
        )
    if normalized_architecture in support.mtp_architectures:
        return RuntimeCompatibilityDecision(
            status=RuntimeCompatibility.SUPPORTED,
            architecture=normalized_architecture,
            runtime_id=support.runtime_id,
            runtime_build=support.runtime_build,
            image_ref=selected_image,
            message=(
                f"Native MTP is supported for {normalized_architecture!r} with "
                f"{nextn_predict_layers} embedded NextN layer(s)."
            ),
        )
    return RuntimeCompatibilityDecision(
        status=RuntimeCompatibility.UNSUPPORTED,
        architecture=normalized_architecture,
        runtime_id=support.runtime_id,
        runtime_build=support.runtime_build,
        image_ref=selected_image,
        message=(
            f"llama.cpp {support.runtime_build} does not advertise native MTP for "
            f"architecture {normalized_architecture!r}."
        ),
    )


def _image_matches_manifest(
    image_ref: str,
    manifest: LlamaCppSupportManifest,
) -> bool:
    accepted = {
        manifest.image_ref,
        *manifest.compatible_image_refs,
        f"{manifest.image_ref}@{manifest.image_digest}",
        f"{manifest.image_ref.split(':', 1)[0]}@{manifest.image_digest}",
    }
    return image_ref in accepted


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = _optional_string(payload.get(key))
    if value is None:
        raise RuntimeError(f"llama.cpp support manifest is missing {key!r}")
    return value


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    return text or None


def _string_set(value: Any) -> frozenset[str]:
    if not isinstance(value, list):
        return frozenset()
    return frozenset(
        text
        for item in value
        if (text := _optional_string(item)) is not None
    )
