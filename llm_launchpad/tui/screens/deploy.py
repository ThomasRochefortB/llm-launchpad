"""Deploy screen: backend selection, model config, deploy options.

Two sub-flows: llama.cpp (ranked GGUF/custom) and vLLM (model params).
Keyboard-driven form navigation with enter-to-proceed.
"""

from __future__ import annotations

from ..widgets.vision_options import VisionOptions

import json

from rich.markup import escape
from textual.actions import SkipAction
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Input,
    OptionList,
    Select,
    Static,
    Switch,
)
from textual.widgets.option_list import Option

from ...core.coerce import positive_int
from ...core.hf_models import ModelCandidate, VllmMemoryBreakdown, fetch_vllm_memory_breakdown
from ...core.inference_options import recommended_vllm_tool_call_parser
from ...core.modal_gpu import ModalGpuSpec, fetch_modal_gpu_catalog
from ...core.naming import (
    auto_instance_name_for_backend,
    build_deployment_name,
    default_served_model_name,
    slugify_instance_name,
)
from ...core.prime_backend import (
    PRIME_VRAM_HEADROOM_FACTOR,
    PrimeBackend,
    is_compatible_prime_offer,
    preferred_prime_offer_image,
)
from ...core.reasoning_profiles import discover_reasoning_capabilities
from ...core.providers import refuse as refuse_deployment
from ...core.vast_runtime import VAST_MAX_GPU_COUNT
from ...core.vast_backend import VastBackend
from ...protocol.enums import BackendType, ComputeProvider
from ...protocol.models import (
    ComputeOffer,
    DeploymentConfig,
    PrimeProviderOptions,
    StorageSnapshot,
    VastOffer,
    VastOfferQuery,
    VastProviderOptions,
)
from ..format import clip
from ..gpu_config import (
    DEFAULT_GPU_COUNT,
    DEFAULT_GPU_TYPE,
    build_gpu_type_options,
    normalize_gpu_type,
    parse_gpu_count,
)
from ..navigation import is_focusable_for_navigation, move_focus_across_option_lists, move_focus_across_widgets
from ..workers import (
    LlamaCppModelsFailed,
    LlamaCppModelsLoaded,
    LlamaCppQuantsFailed,
    LlamaCppQuantsLoaded,
    StorageFailed,
    StorageLoaded,
    VllmModelsFailed,
    VllmModelsLoaded,
)
from ..widgets.input_form import FormField, ToggleField
from .copy_enabled import CopyEnabledScreen


_MODEL_LOOKUP_DEBOUNCE_SECONDS = 0.35

_RANKING_SUBTITLES: dict[str, dict[str, str]] = {
    BackendType.LLAMACPP: {
        "cached": "models cached in your storage volumes",
        "downloads": "top 10 GGUF text-generation models on Hugging Face",
        "trending": "trending GGUF text-generation models on Hugging Face",
    },
    BackendType.VLLM: {
        "cached": "models cached in your storage volumes",
        "downloads": "top 10 text-generation models on Hugging Face",
        "trending": "trending text-generation models on Hugging Face",
    },
}


def _ranking_subtitle(backend: BackendType, mode: str) -> str:
    subtitles = _RANKING_SUBTITLES.get(backend) or {}
    return subtitles.get(mode) or "models cached in your storage volumes"


_GPU_PANEL_SUBTITLES: dict[str, dict[ComputeProvider, str]] = {
    BackendType.LLAMACPP: {
        ComputeProvider.MODAL: "Select a Modal GPU shape.",
        ComputeProvider.PRIME: "GPU shape and count come from the Prime offer bound above.",
        ComputeProvider.VAST: "GPU shape comes from the Vast.ai rental bound above.",
    },
    BackendType.VLLM: {
        ComputeProvider.MODAL: (
            "Choose deployment GPUs and in-replica tensor sharding. "
            "Base Modal hourly price per GPU is shown when available."
        ),
        ComputeProvider.PRIME: (
            "GPU shape and count come from the Prime offer bound above; "
            "tensor sharding follows its GPU count."
        ),
        ComputeProvider.VAST: (
            "GPU shape comes from the Vast.ai rental bound above, so tensor "
            "sharding matches the rented device count."
        ),
    },
}


def _gpu_panel_subtitle(backend: BackendType, provider: ComputeProvider) -> str:
    """Return GPU-panel help text that matches the selected compute provider."""
    by_provider = _GPU_PANEL_SUBTITLES[backend]
    return by_provider.get(provider, by_provider[ComputeProvider.MODAL])


def _set_option_list(option_list: OptionList, options: list[Option]) -> None:
    """Replace an option list's contents, hiding it while it has none.

    An empty ``OptionList`` still paints its border, so a picker that has not
    loaded yet reads as a broken empty box and costs three rows of a short
    terminal. Hiding it through the shared ``hidden`` class also keeps it out
    of keyboard navigation until it holds something selectable.
    """
    option_list.set_options(options)
    option_list.set_class(not options, "hidden")


_MODEL_COLUMN_WIDTH = 38


def _model_row(repo_id: str, detail: str) -> str:
    """Render a model picker row with a stable repo-id column."""
    return f"  {clip(repo_id, _MODEL_COLUMN_WIDTH):<{_MODEL_COLUMN_WIDTH}} {detail}"


def _format_hourly_cost(value: float | None) -> str:
    if value is None:
        return "price n/a"
    return f"${value:.2f}/hr"


def _format_always_on_monthly_cost(value: float | None) -> str:
    if value is None:
        return "monthly est. n/a"
    return f"~${value * 24 * 30:,.0f}/mo at 24/7"


def _is_plausible_model_lookup(value: str) -> bool:
    """Return whether a partial model value is worth resolving remotely."""
    normalized = value.strip()
    owner, separator, model = normalized.partition("/")
    return bool(
        separator
        and owner
        and model
        and not any(character.isspace() for character in normalized)
    )


class GpuTypesLoaded(Message):
    """GPU type options were fetched successfully."""

    def __init__(self, gpu_types: list[ModalGpuSpec]) -> None:
        super().__init__()
        self.gpu_types = gpu_types


class GpuTypesFailed(Message):
    """GPU type option fetch failed."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class PrimeOffersLoaded(Message):
    """Prime availability was fetched successfully."""

    def __init__(self, offers: list[ComputeOffer]) -> None:
        super().__init__()
        self.offers = offers


class PrimeOffersFailed(Message):
    """Prime availability fetch failed."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class VastOffersLoaded(Message):
    """Vast.ai availability was fetched successfully."""

    def __init__(self, offers: list[VastOffer]) -> None:
        super().__init__()
        self.offers = offers


class VastOffersFailed(Message):
    """Vast.ai availability fetch failed."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class VllmMemoryLoaded(Message):
    """Heuristic vLLM memory estimate fetched for a model."""

    def __init__(self, repo_id: str, revision: str | None, estimate: VllmMemoryBreakdown | None) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.revision = revision
        self.estimate = estimate


class VllmMemoryFailed(Message):
    """Heuristic vLLM memory fetch failed."""

    def __init__(self, repo_id: str, revision: str | None, error: str) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.revision = revision
        self.error = error


def _is_focusable_for_arrow_navigation(widget: Widget) -> bool:
    return is_focusable_for_navigation(widget, check_hidden_ancestor=True)


def _cached_models_from_snapshot(snapshot: StorageSnapshot, backend: BackendType) -> list[ModelCandidate]:
    rows = snapshot.llamacpp_models if backend == BackendType.LLAMACPP else snapshot.vllm_models
    repo_by_key: dict[str, str] = {}
    size_by_key: dict[str, int] = {}
    quants_by_key: dict[str, set[str]] = {}

    for row in rows:
        repo_id = row.model_id.strip()
        if not repo_id:
            continue
        key = repo_id.casefold()
        if key not in repo_by_key:
            repo_by_key[key] = repo_id
        size_by_key[key] = size_by_key.get(key, 0) + max(0, row.size_bytes)
        if backend == BackendType.LLAMACPP:
            quant = (row.quant or "").strip().upper()
            if quant:
                quants_by_key.setdefault(key, set()).add(quant)

    sorted_keys = sorted(repo_by_key, key=lambda key: (-size_by_key.get(key, 0), repo_by_key[key].casefold()))
    return [
        ModelCandidate(
            repo_id=repo_by_key[key],
            quantizations=tuple(sorted(quants_by_key.get(key, set()))) if backend == BackendType.LLAMACPP else (),
        )
        for key in sorted_keys
    ]


def _model_from_option_id(option_id: str, ranked_models: list[ModelCandidate]) -> ModelCandidate | None:
    if not option_id.startswith("model-"):
        return None
    try:
        idx = int(option_id.split("-", 1)[1])
    except ValueError:
        return None
    if idx < 0 or idx >= len(ranked_models):
        return None
    return ranked_models[idx]


def _normalize_vram_map(vram_gb_by_quant: dict[str, float] | None) -> dict[str, float]:
    if not isinstance(vram_gb_by_quant, dict):
        return {}
    normalized: dict[str, float] = {}
    for quant, value in vram_gb_by_quant.items():
        quant_key = str(quant).strip().upper()
        if not quant_key:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric <= 0:
            continue
        current = normalized.get(quant_key)
        if current is None or numeric > current:
            normalized[quant_key] = numeric
    return normalized


def _format_vram_gb(vram_gb: float) -> str:
    return f"{vram_gb:.1f} GB"


def _format_quant_with_vram(quant: str, vram_gb_by_quant: dict[str, float]) -> str:
    vram_gb = vram_gb_by_quant.get(quant.strip().upper())
    if vram_gb is None:
        return quant
    return f"{quant} (~{_format_vram_gb(vram_gb)})"


def _compatible_prime_offers(
    offers: list[ComputeOffer],
    required_vram_gb: float | None,
    backend: BackendType = BackendType.LLAMACPP,
) -> list[ComputeOffer]:
    """Return fixed-price GPU offers that fit one model requirement."""

    required_image = preferred_prime_offer_image(backend)
    compatible = [
        offer
        for offer in offers
        if is_compatible_prime_offer(
            offer,
            required_vram_gb,
            required_image=required_image,
        )
    ]
    compatible.sort(
        key=lambda offer: (
            offer.price_per_hour is None,
            offer.price_per_hour
            if offer.price_per_hour is not None
            else float("inf"),
            offer.gpu_count,
            offer.id,
        )
    )
    return compatible


def _prime_offer_options(
    offers: list[ComputeOffer],
) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    for offer in offers:
        location = offer.country or offer.region or offer.data_center or "unknown location"
        price = (
            f"${offer.price_per_hour:.3f}/hr"
            if offer.price_per_hour is not None
            else "price n/a"
        )
        memory = (
            f" · {offer.gpu_memory_gb:.0f} GB VRAM"
            if offer.gpu_memory_gb is not None
            else ""
        )
        # Price leads because that is what these rows are compared on; the
        # opaque offer id is last and labelled, so it reads as a reference
        # rather than as the name of the thing being chosen.
        options.append(
            (
                f"{price} · {offer.gpu_count}x {offer.gpu_type}{memory} · "
                f"in {location} · offer {clip(offer.id, 8)}",
                offer.id,
            )
        )
    return options


def _prime_offer_status(
    offer_count: int,
    required_vram_gb: float | None,
    backend: BackendType = BackendType.LLAMACPP,
) -> str:
    # The provider's image identifier ("ubuntu_22_cuda_12") is an internal
    # detail; the user is choosing capacity, not a base image.
    strategy = "secure on-demand"
    if required_vram_gb is None:
        if offer_count:
            return (
                f"[dim]{offer_count} live {strategy} GPU offers. "
                "Model VRAM is not known yet; "
                "choose a model to narrow them.[/dim]"
            )
        return (
            f"[yellow]No {strategy} GPU offers are currently available.[/yellow]"
        )
    required_with_headroom = required_vram_gb * PRIME_VRAM_HEADROOM_FACTOR
    if offer_count:
        return (
            f"[dim]{offer_count} live {strategy} GPU offers fit this model's "
            f"~{required_with_headroom:.1f} GB requirement.[/dim]"
        )
    return (
        f"[yellow]No live {strategy} GPU offer has enough memory for this model's "
        f"~{required_with_headroom:.1f} GB requirement.[/yellow]"
    )


# A Vast rental re-quotes its offer at deploy time, so the picker must only
# present shapes the deployment path can actually deliver.
VAST_VRAM_HEADROOM_FACTOR = 1.05
DEFAULT_VAST_DISK_GB = 100


def _compatible_vast_offers(
    offers: list[VastOffer],
    required_vram_gb: float | None,
) -> list[VastOffer]:
    """Return priced rentals whose combined GPUs fit one model requirement."""
    required = (required_vram_gb or 0.0) * VAST_VRAM_HEADROOM_FACTOR
    fitting = [
        offer
        for offer in offers
        if 1 <= offer.gpu_count <= VAST_MAX_GPU_COUNT
        and offer.costs.total_per_hour_usd is not None
        and offer.gpu_memory_gib * offer.gpu_count >= required
    ]
    # At equal price prefer the simpler topology: fewer devices means fewer
    # ways for a marketplace host to disagree with the memory plan.
    return sorted(fitting, key=lambda offer: (offer.costs.total_per_hour_usd or 0.0, offer.gpu_count))


def _vast_offer_options(offers: list[VastOffer]) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    for offer in offers:
        price = offer.costs.total_per_hour_usd
        price_text = f"${price:.3f}/hr" if price is not None else "price n/a"
        location = offer.location or "unknown location"
        total = offer.gpu_memory_gb * offer.gpu_count
        memory = (
            f"{offer.gpu_memory_gb:.0f} GB each ({total:.0f} GB total)"
            if offer.gpu_count > 1
            else f"{offer.gpu_memory_gb:.0f} GB VRAM"
        )
        options.append(
            (
                f"{price_text} incl. disk · {offer.gpu_count}x {offer.gpu_type} · "
                f"{memory} · in {location} · offer {clip(offer.id, 8)}",
                offer.id,
            )
        )
    return options


def _vast_offer_status(offer_count: int, required_vram_gb: float | None) -> str:
    if required_vram_gb is None:
        if offer_count:
            return (
                f"[dim]{offer_count} live rentals. Prices include disk; "
                "network traffic is billed separately. Choose a model to narrow "
                "them.[/dim]"
            )
        return "[yellow]No live Vast.ai rentals are currently available.[/yellow]"
    required_with_headroom = required_vram_gb * VAST_VRAM_HEADROOM_FACTOR
    if offer_count:
        return (
            f"[dim]{offer_count} live rentals fit this model's "
            f"~{required_with_headroom:.1f} GB requirement. Prices include disk; "
            "network traffic is billed separately.[/dim]"
        )
    return (
        "[yellow]No live Vast.ai rental has enough memory for this "
        f"model's ~{required_with_headroom:.1f} GB requirement.[/yellow]"
    )


def _advance_deploy_focus(screen: CopyEnabledScreen, navigation_order: tuple[str, ...]) -> None:
    """Advance focus to the next visible deploy form widget."""
    move_focus_across_widgets(
        screen,
        navigation_order,
        direction=1,
        is_focusable=_is_focusable_for_arrow_navigation,
    )


class _OptionListArrowNavigationMixin:
    """Arrow keys walk a screen's option lists first, then its focus order."""

    OPTION_LIST_IDS: tuple[str, ...] = ()
    NAVIGATION_ORDER: tuple[str, ...] = ()

    def _navigate_option_lists(self, direction: int) -> None:
        if move_focus_across_option_lists(self, self.OPTION_LIST_IDS, direction=direction):
            return
        if move_focus_across_widgets(
            self,
            self.NAVIGATION_ORDER,
            direction=direction,
            is_focusable=_is_focusable_for_arrow_navigation,
        ):
            return
        raise SkipAction()

    def action_navigate_option_list_down(self) -> None:
        self._navigate_option_lists(1)

    def action_navigate_option_list_up(self) -> None:
        self._navigate_option_lists(-1)


class _CostPreviewMixin:
    """Shared hourly cost preview behavior for custom deploy forms."""

    _gpu_price_by_value: dict[str, float]
    _provider: ComputeProvider
    _prime_offers: dict[str, ComputeOffer]
    _selected_prime_offer_id: str | None
    _selected_gpu_type: str | None
    _vast_offers: dict[str, VastOffer]
    _selected_vast_offer_id: str | None

    def _current_gpu_hourly_price(self) -> float | None:
        if self._provider == ComputeProvider.PRIME:
            offer = self._prime_offers.get(self._selected_prime_offer_id or "")
            return offer.price_per_hour if offer is not None else None
        if self._provider == ComputeProvider.VAST:
            vast_offer = self._vast_offers.get(self._selected_vast_offer_id or "")
            return vast_offer.costs.total_per_hour_usd if vast_offer is not None else None
        return self._gpu_price_by_value.get((self._selected_gpu_type or "").strip())

    def _build_vast_provider_options(self, disk_field_id: str) -> VastProviderOptions | None:
        """Validate the bound Vast rental and approve its quoted hourly total.

        The deploy path re-quotes the offer and refuses anything above the
        approved price, so the cap is exactly what the picker showed: the user
        approves the number they read, not a padded one.
        """
        from textual.widgets import Input

        offer = self._vast_offers.get(self._selected_vast_offer_id or "")
        if offer is None:
            self.app.notify("Select a live Vast.ai rental.", severity="error", timeout=5)
            return None
        price = offer.costs.total_per_hour_usd
        if price is None or price <= 0:
            self.app.notify(
                "The selected Vast.ai rental has no quoted hourly price. Refresh rentals.",
                severity="error",
                timeout=6,
            )
            return None
        disk_gb = positive_int(self.query_one(disk_field_id, Input).value)
        if disk_gb is None:
            self.app.notify("Vast disk size must be an integer >= 1 GB.", severity="error", timeout=5)
            return None
        return VastProviderOptions(
            offer_id=offer.id,
            disk_gb=disk_gb,
            max_hourly_cost_usd=price,
            machine_id=offer.machine_id,
            # The user approved this topology at this price; a rental that no
            # longer matches it is refused rather than silently substituted.
            gpu_count=offer.gpu_count,
        )

    def _update_cost_preview(self, preview_id: str) -> None:
        from textual.widgets import Static

        try:
            preview = self.query_one(f"#{preview_id}", Static)
        except Exception:
            return
        price = self._current_gpu_hourly_price()
        if price is None:
            preview.update("[dim]Hourly: price n/a[/dim]")
            return
        preview.update(
            f"[dim]Hourly: {_format_hourly_cost(price)} · "
            f"{_format_always_on_monthly_cost(price)}[/dim]"
        )


class BackendSelectScreen(CopyEnabledScreen):
    """Step 1: pick backend (llama.cpp or vLLM)."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="screen-scroll"):
            yield Static("[bold #7bf168]Advanced deploy[/]  [dim]Step 1: Choose serving engine[/dim]")
            yield Static("")
            yield OptionList(
                Option(
                    "  [bold]llama.cpp (GGUF)[/bold]  [dim]· recommended[/dim]\n"
                    "  [dim]Quantized single-file models. Smaller GPUs, faster cold starts,\n"
                    "  one request at a time.[/dim]",
                    id="llamacpp",
                ),
                Option(
                    "  [bold]vLLM[/bold]\n"
                    "  [dim]Full-precision Hugging Face models. Higher throughput under\n"
                    "  concurrency, tensor parallelism across GPUs.[/dim]",
                    id="vllm",
                ),
                id="backend-list",
            )
        yield Footer()

    def on_mount(self) -> None:
        backend_list = self.query_one("#backend-list", OptionList)
        if backend_list.option_count > 0:
            backend_list.highlighted = 0
        backend_list.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id == "llamacpp":
            self.app.push_screen(LlamaCppDeployScreen())
        elif event.option.id == "vllm":
            self.app.push_screen(VllmDeployScreen())

    def action_pop_screen(self) -> None:
        self.app.pop_screen()


class LlamaCppDeployScreen(_OptionListArrowNavigationMixin, _CostPreviewMixin, CopyEnabledScreen):
    """llama.cpp deploy form."""

    BINDINGS = [
        Binding("up", "navigate_option_list_up", show=False, priority=True),
        Binding("down", "navigate_option_list_down", show=False, priority=True),
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+d", "do_deploy", "Deploy", show=True, priority=True),
        Binding("ctrl+s", "open_storage", "Storage", show=True),
        Binding("p", "predownload_highlighted", "Pre-download", show=True),
    ]
    _MODEL_LIST_ID = "llama-model-list"
    OPTION_LIST_IDS = ("llama-rank-mode", "llama-model-list", "llama-quant-list")
    NAVIGATION_ORDER = (
        "llama-rank-mode",
        "llama-model-list",
        "repo-id",
        "quant",
        "llama-quant-list",
        "provider-llama",
        "prime-offer-llama",
        "vast-offer-llama",
        "gpu-type-llama",
        "gpu-count-llama",
        "vision-mode",
        "projector-repo",
        "projector-revision",
        "projector-file",
        "toggle-advanced-llama",
        "warmup",
        "revision",
        "prime-auto-disk-llama",
        "prime-disk-id-llama",
        "prime-insecure-http-llama",
        "prime-keep-failed-llama",
        "vast-disk-llama",
        "server-args",
        "host-input",
        "port-input",
        "n-gpu-layers",
        "llama-image-no-cache",
        "show-debug-logs-llama",
        "instance-name-llama",
        "app-name-llama",
        "deploy-btn",
    )

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="screen-scroll"):
            yield Static("[bold #7bf168]Advanced deploy llama.cpp[/]  [dim]Step 2: Model & options[/dim]")
            yield Static("")

            yield Static(
                "[bold]Model[/bold]  [dim](cached models in your storage volumes)[/dim]",
                id="llama-model-ranking-title",
            )
            yield OptionList(
                Option("  Cached in storage", id="rank-cached"),
                Option("  Most downloaded", id="rank-downloads"),
                Option("  Trending", id="rank-trending"),
                id="llama-rank-mode",
            )
            yield Static("[dim]Loading model suggestions...[/dim]", id="llama-model-status")
            yield OptionList(id="llama-model-list")

            yield FormField(
                "Hugging Face repo-id",
                "repo-id",
                hint="e.g., Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            )
            yield FormField("Quant pattern", "quant", default="Q4_K_M")
            yield Static("[dim]Quantizations: enter repo-id to detect GGUF variants[/dim]", id="llama-quant-status")
            yield OptionList(id="llama-quant-list")
            yield Static("")

            yield Static("Compute provider", classes="form-label")
            yield Select(
                options=[
                    ("Modal", "modal"),
                    ("Prime Intellect", "prime"),
                    ("Vast.ai", "vast"),
                ],
                value="modal",
                allow_blank=False,
                id="provider-llama",
            )
            yield Static("Prime GPU offer", classes="form-label prime-only")
            yield Select(
                options=[],
                prompt="Select an exact live Prime offer",
                id="prime-offer-llama",
                classes="prime-only",
            )
            yield Static(
                "[dim]Prime offers are secure, on-demand availability sorted by price.[/dim]",
                id="prime-offer-status-llama",
                classes="prime-only",
            )
            yield Static("Vast.ai rental", classes="form-label vast-only")
            yield Select(
                options=[],
                prompt="Select a live Vast.ai rental",
                id="vast-offer-llama",
                classes="vast-only",
            )
            yield Static(
                "[dim]Vast.ai rentals are priced including disk.[/dim]",
                id="vast-offer-status-llama",
                classes="vast-only",
            )

            # Options
            with Vertical(classes="gpu-config-panel"):
                yield Static("GPU configuration", classes="form-section-title")
                yield Static(
                    "Select a Modal GPU shape.",
                    id="gpu-config-subtitle-llama",
                    classes="form-section-subtitle",
                )
                with Horizontal(id="gpu-config-row-llama", classes="gpu-config-main-row"):
                    with Vertical(id="gpu-type-group-llama"):
                        yield Static("GPU type", classes="form-label")
                        yield Select(
                            options=[(DEFAULT_GPU_TYPE, DEFAULT_GPU_TYPE)],
                            prompt="Select GPU type",
                            value=DEFAULT_GPU_TYPE,
                            id="gpu-type-llama",
                        )
                    with Vertical(id="gpu-count-group-llama"):
                        yield Static("GPU count", classes="form-label")
                        yield Input(
                            value=str(DEFAULT_GPU_COUNT),
                            placeholder="1",
                            id="gpu-count-llama",
                            type="integer",
                        )
                yield Static("", id="llama-cost-preview")

            yield Static("")

            yield VisionOptions(BackendType.LLAMACPP)

            # Advanced options (collapsed by default)
            yield Button("Advanced options...", id="toggle-advanced-llama", variant="default")
            yield ToggleField("Warm up after deploy", "warmup", default=True, classes="llama-advanced")
            yield FormField("HF revision (optional)", "revision", classes="llama-advanced")
            yield ToggleField(
                "Attach persistent cache disk",
                "prime-auto-disk-llama",
                default=True,
                classes="llama-advanced prime-only",
            )
            yield FormField(
                "Prime disk ID (optional)",
                "prime-disk-id-llama",
                hint="Leave blank to auto-attach a persistent cache disk",
                classes="llama-advanced prime-only",
            )
            yield ToggleField(
                "Use direct HTTP fallback (insecure)",
                "prime-insecure-http-llama",
                default=False,
                classes="llama-advanced prime-only",
            )
            yield ToggleField(
                "Keep failed Prime pod (billing may continue)",
                "prime-keep-failed-llama",
                default=False,
                classes="llama-advanced prime-only",
            )
            yield FormField(
                "Vast disk size (GB)",
                "vast-disk-llama",
                default=str(DEFAULT_VAST_DISK_GB),
                input_type="integer",
                hint="Rented alongside the GPU and included in the quoted price",
                classes="llama-advanced vast-only",
            )
            yield Static("Runtime", classes="form-group-heading llama-advanced")
            yield FormField(
                "Server args",
                "server-args",
                hint="e.g., --ctx-size 65536",
                classes="llama-advanced",
            )
            yield FormField("Host", "host-input", default="0.0.0.0", classes="llama-advanced")
            yield FormField("Port", "port-input", default="8080", classes="llama-advanced")
            yield FormField(
                "n_gpu_layers (blank=auto)",
                "n-gpu-layers",
                classes="llama-advanced",
            )
            yield ToggleField(
                "Force fresh llama.cpp image pull/build (ignore cache)",
                "llama-image-no-cache",
                default=False,
                classes="llama-advanced",
            )
            yield ToggleField(
                "Show debug logs (full raw backend logs)",
                "show-debug-logs-llama",
                default=False,
                classes="llama-advanced",
            )

            yield Static("")
            yield Static("Naming", classes="form-group-heading llama-advanced")
            yield FormField(
                "Instance name (optional)",
                "instance-name-llama",
                hint="Auto-derived from repo if blank",
                classes="llama-advanced",
            )
            yield FormField(
                "App name override (optional)",
                "app-name-llama",
                hint="Advanced: explicit provider resource name",
                classes="llama-advanced",
            )
            yield Static(
                "[dim]App name preview: auto[/dim]",
                id="llama-app-preview",
                classes="llama-advanced",
            )
            yield Static("")
            yield Button("Deploy", id="deploy-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self._rank_mode = "cached"
        self._ranked_models: list[ModelCandidate] = []
        self._cached_models: list[ModelCandidate] = []
        self._cached_repo_to_quants: dict[str, tuple[str, ...]] = {}
        self._has_cached_snapshot = False
        self._repo_to_quants: dict[str, tuple[str, ...]] = {}
        self._repo_to_quant_vram: dict[str, dict[str, float]] = {}
        self._repo_to_architecture: dict[tuple[str, str], str | None] = {}
        self._repo_to_compatibility: dict[
            tuple[str, str], tuple[str, str, str | None]
        ] = {}
        self._last_quant_lookup: tuple[str, str] | None = None
        self._quant_lookup_timer: Timer | None = None
        self._updating_quant_input = False
        self._quant_touched = False
        self._selected_gpu_type = DEFAULT_GPU_TYPE
        # Binding a Prime offer overwrites the GPU dropdown with that offer's
        # GPU. Remember the Modal choice so switching back does not leave a
        # Prime-only GPU name selected for a Modal deploy.
        self._last_modal_gpu_type = DEFAULT_GPU_TYPE
        self._modal_gpu_values: set[str] = {DEFAULT_GPU_TYPE}
        self._provider = ComputeProvider.MODAL
        self._prime_offers: dict[str, ComputeOffer] = {}
        self._selected_prime_offer_id: str | None = None
        self._vast_offers: dict[str, VastOffer] = {}
        self._selected_vast_offer_id: str | None = None
        self._gpu_price_by_value: dict[str, float] = {}
        self._rank_mode_touched = False
        self._focus_model_list_when_loaded = False
        for list_id in ("#llama-model-list", "#llama-quant-list"):
            self.query_one(list_id, OptionList).add_class("hidden")
        for widget in self.query(".llama-advanced"):
            widget.add_class("hidden")
        for widget in self.query(".prime-only"):
            widget.add_class("hidden")
        for widget in self.query(".vast-only"):
            widget.add_class("hidden")
        rank_mode_list = self.query_one("#llama-rank-mode", OptionList)
        if rank_mode_list.option_count > 0:
            rank_mode_list.highlighted = 0
        rank_mode_list.focus()
        self._set_ranking_title()
        self._refresh_gpu_types()
        self._set_model_status("[dim]Loading cached models from storage...[/dim]")
        self._refresh_cached_models_from_storage()
        self._refresh_app_preview()
        self._update_cost_preview("llama-cost-preview")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "llama-rank-mode":
            selected_mode = self._resolve_rank_mode(event.option.id or "")
            if selected_mode is None:
                return
            self._rank_mode_touched = True
            if selected_mode != self._rank_mode:
                self._rank_mode = selected_mode
                self._set_ranking_title()
                self._ranked_models = []
                self._set_model_status("[dim]Loading model suggestions...[/dim]")
                _set_option_list(self.query_one("#llama-model-list", OptionList), [])
                if self._rank_mode == "cached":
                    self._set_model_status("[dim]Loading cached models from storage...[/dim]")
                    if self._has_cached_snapshot:
                        self._show_cached_models()
                    self._refresh_cached_models_from_storage()
                else:
                    self.app.begin_fetch_llamacpp_models(self._rank_mode, self)  # type: ignore[attr-defined]
            model_list = self.query_one("#llama-model-list", OptionList)
            # The picker is hidden until it has rows, so a still-loading list
            # cannot take focus yet. Hand it over once the rows land.
            self._focus_model_list_when_loaded = model_list.option_count == 0
            _advance_deploy_focus(self, self.NAVIGATION_ORDER)
            return

        if event.option_list.id == "llama-model-list":
            self._apply_ranked_model_selection(event.option.id or "")
            self._refresh_app_preview()
            _advance_deploy_focus(self, self.NAVIGATION_ORDER)
            return

        if event.option_list.id == "llama-quant-list":
            self._apply_quant_selection(event.option.id or "")
            _advance_deploy_focus(self, self.NAVIGATION_ORDER)
            return

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Mirror highlighted model into repo-id for keyboard-first flow."""
        if event.option_list.id != "llama-model-list":
            return
        self._apply_ranked_model_selection(event.option.id or "")
        self._refresh_app_preview()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "toggle-advanced-llama":
            for widget in self.query(".llama-advanced"):
                widget.toggle_class("hidden")
            self._sync_prime_visibility()
        elif event.button.id == "deploy-btn":
            self._do_deploy()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "quant" and not self._updating_quant_input:
            self._quant_touched = True
        if event.input.id in {"repo-id", "instance-name-llama", "app-name-llama"}:
            self._refresh_app_preview()
        if event.input.id in {"repo-id", "revision"}:
            self._schedule_quantization_lookup()
        if event.input.id in {"repo-id", "quant"} and self._prime_offers:
            self._refresh_prime_offer_options()
        if event.input.id == "gpu-count-llama":
            self._update_cost_preview("llama-cost-preview")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "provider-llama":
            if not isinstance(event.value, str):
                return
            self._provider = ComputeProvider(event.value)
            self._sync_prime_visibility()
            if self._provider == ComputeProvider.PRIME and not self._prime_offers:
                self._refresh_prime_offers()
            elif self._provider == ComputeProvider.VAST:
                if self._vast_offers:
                    self._refresh_vast_offer_options()
                else:
                    self._refresh_vast_offers()
            elif self._provider == ComputeProvider.MODAL:
                self._selected_gpu_type = self._last_modal_gpu_type
                self._selected_prime_offer_id = None
                self._selected_vast_offer_id = None
                self._refresh_gpu_types()
            self._refresh_app_preview()
            self._update_cost_preview("llama-cost-preview")
            return
        if event.select.id == "vast-offer-llama":
            if not isinstance(event.value, str):
                return
            self._selected_vast_offer_id = event.value
            offer = self._vast_offers.get(event.value)
            if offer is not None:
                self._selected_gpu_type = offer.gpu_type
                gpu_type_select = self.query_one("#gpu-type-llama", Select)
                gpu_type_select.set_options([(offer.gpu_type, offer.gpu_type)])
                gpu_type_select.value = offer.gpu_type
                self.query_one("#gpu-count-llama", Input).value = str(offer.gpu_count)
            self._update_cost_preview("llama-cost-preview")
            return
        if event.select.id == "prime-offer-llama":
            if not isinstance(event.value, str):
                return
            self._selected_prime_offer_id = event.value
            offer = self._prime_offers.get(event.value)
            if offer is not None:
                self._selected_gpu_type = offer.gpu_type
                self.query_one("#gpu-type-llama", Select).set_options(
                    [(offer.gpu_type, offer.gpu_type)]
                )
                self.query_one("#gpu-type-llama", Select).value = offer.gpu_type
                self.query_one("#gpu-count-llama", Input).value = str(offer.gpu_count)
            self._update_cost_preview("llama-cost-preview")
            return
        if event.select.id != "gpu-type-llama":
            return
        if isinstance(event.value, str) and event.value.strip():
            self._selected_gpu_type = normalize_gpu_type(event.value)
            if self._selected_gpu_type in self._modal_gpu_values:
                self._last_modal_gpu_type = self._selected_gpu_type
        self._update_cost_preview("llama-cost-preview")

    def _sync_prime_visibility(self) -> None:
        self.query_one("#gpu-config-subtitle-llama", Static).update(
            _gpu_panel_subtitle(BackendType.LLAMACPP, self._provider)
        )
        # A bound Prime offer dictates the GPU shape and count, and the deploy
        # path overwrites whatever these hold. Leaving them editable invites
        # changes that are silently discarded.
        offer_bound = self._provider in (ComputeProvider.PRIME, ComputeProvider.VAST)
        for field_id in ("#gpu-type-llama", "#gpu-count-llama",):
            self.query_one(field_id).disabled = offer_bound
        advanced_visible = not self.query_one("#warmup").has_class("hidden")
        for provider_class, provider in (
            (".prime-only", ComputeProvider.PRIME),
            (".vast-only", ComputeProvider.VAST),
        ):
            for widget in self.query(provider_class):
                hide = self._provider != provider
                if widget.has_class("llama-advanced") and not advanced_visible:
                    hide = True
                widget.set_class(hide, "hidden")

    def _refresh_prime_offers(self) -> None:
        self.query_one("#prime-offer-status-llama", Static).update(
            "[dim]Loading Prime offers...[/dim]"
        )
        self.run_worker(
            self._run_fetch_prime_offers,
            name="llamacpp-fetch-prime-offers",
            thread=True,
        )

    def _run_fetch_prime_offers(self) -> None:
        try:
            offers = PrimeBackend().list_offers()
        except Exception as exc:
            self.post_message(PrimeOffersFailed(str(exc)))
            return
        self.post_message(PrimeOffersLoaded(offers))

    def on_prime_offers_loaded(self, message: PrimeOffersLoaded) -> None:
        self._prime_offers = {offer.id: offer for offer in message.offers}
        self._refresh_prime_offer_options()

    def _current_llamacpp_required_vram(self) -> float | None:
        repo_key = self.query_one("#repo-id", Input).value.strip().casefold()
        quant_key = self.query_one("#quant", Input).value.strip().upper()
        if not repo_key or not quant_key:
            return None
        return self._repo_to_quant_vram.get(repo_key, {}).get(quant_key)

    def _refresh_prime_offer_options(self) -> None:
        required_vram_gb = self._current_llamacpp_required_vram()
        offers = _compatible_prime_offers(
            list(self._prime_offers.values()),
            required_vram_gb,
            BackendType.LLAMACPP,
        )
        options = _prime_offer_options(offers)
        selector = self.query_one("#prime-offer-llama", Select)
        selector.set_options(options)
        status = self.query_one("#prime-offer-status-llama", Static)
        if options:
            self._selected_prime_offer_id = options[0][1]
            selector.value = options[0][1]
        else:
            self._selected_prime_offer_id = None
        status.update(
            _prime_offer_status(
                len(options),
                required_vram_gb,
                BackendType.LLAMACPP,
            )
        )

    def on_prime_offers_failed(self, message: PrimeOffersFailed) -> None:
        self.query_one("#prime-offer-status-llama", Static).update(
            f"[red]Could not load Prime offers:[/red] {escape(message.error)}"
        )

    def _refresh_vast_offers(self) -> None:
        self.query_one("#vast-offer-status-llama", Static).update(
            "[dim]Loading Vast.ai rentals...[/dim]"
        )
        self.run_worker(
            self._run_fetch_vast_offers,
            name="llamacpp-fetch-vast-offers",
            thread=True,
        )

    def _run_fetch_vast_offers(self) -> None:
        try:
            offers = VastBackend().list_offers(
                VastOfferQuery(gpu_count=None, disk_gb=self._vast_disk_gb())
            )
        except Exception as exc:
            self.post_message(VastOffersFailed(str(exc)))
            return
        self.post_message(VastOffersLoaded(offers))

    def on_vast_offers_loaded(self, message: VastOffersLoaded) -> None:
        self._vast_offers = {offer.id: offer for offer in message.offers}
        self._refresh_vast_offer_options()

    def on_vast_offers_failed(self, message: VastOffersFailed) -> None:
        self.query_one("#vast-offer-status-llama", Static).update(
            f"[red]Could not load Vast.ai rentals:[/red] {escape(message.error)}"
        )

    def _vast_disk_gb(self) -> int:
        return positive_int(self.query_one("#vast-disk-llama", Input).value) or DEFAULT_VAST_DISK_GB

    def _refresh_vast_offer_options(self) -> None:
        required_vram_gb = self._current_llamacpp_required_vram()
        offers = _compatible_vast_offers(list(self._vast_offers.values()), required_vram_gb)
        options = _vast_offer_options(offers)
        selector = self.query_one("#vast-offer-llama", Select)
        selector.set_options(options)
        if options:
            self._selected_vast_offer_id = options[0][1]
            selector.value = options[0][1]
        else:
            self._selected_vast_offer_id = None
        self.query_one("#vast-offer-status-llama", Static).update(
            _vast_offer_status(len(options), required_vram_gb)
        )

    def _refresh_gpu_types(self) -> None:
        self.run_worker(
            lambda: self._run_fetch_gpu_types(),
            name="llama-fetch-gpu-types",
            thread=True,
        )

    def _run_fetch_gpu_types(self) -> None:
        poster = getattr(self, "post_message", None)
        if poster is None:
            return
        try:
            gpu_types = fetch_modal_gpu_catalog()
        except Exception as exc:
            poster(GpuTypesFailed(error=str(exc)))
            return
        poster(GpuTypesLoaded(gpu_types=gpu_types))

    def on_gpu_types_loaded(self, message: GpuTypesLoaded) -> None:
        dropdown = self.query_one("#gpu-type-llama", Select)
        options = build_gpu_type_options(message.gpu_types)
        option_values = [value for _, value in options]
        self._modal_gpu_values = set(option_values)
        if self._selected_gpu_type and self._selected_gpu_type not in option_values:
            options.insert(0, (self._selected_gpu_type, self._selected_gpu_type))
            option_values.insert(0, self._selected_gpu_type)
        if not options:
            return
        dropdown.set_options(options)
        selected = self._selected_gpu_type if self._selected_gpu_type in option_values else option_values[0]
        dropdown.value = selected
        self._selected_gpu_type = selected
        self._gpu_price_by_value = {
            spec.value.strip(): spec.price_per_hour_usd
            for spec in message.gpu_types
            if spec.price_per_hour_usd is not None
        }
        self._update_cost_preview("llama-cost-preview")

    def on_gpu_types_failed(self, _: GpuTypesFailed) -> None:
        # Keep default value if Modal docs fetch fails.
        return

    def on_llama_cpp_models_loaded(self, message: LlamaCppModelsLoaded) -> None:
        if message.mode != self._rank_mode:
            return
        self._ranked_models = message.models
        self._repo_to_quants = {model.repo_id.casefold(): model.quantizations for model in message.models}
        model_list = self.query_one("#llama-model-list", OptionList)
        options = []
        for idx, model in enumerate(message.models):
            downloads = f"{model.downloads:,}" if model.downloads is not None else "-"
            likes = f"{model.likes:,}" if model.likes is not None else "-"
            label = _model_row(model.repo_id, f"downloads={downloads:<10} likes={likes}")
            options.append(Option(label, id=f"model-{idx}"))
        _set_option_list(model_list, options)

        if message.models:
            mode_label = "Most downloaded" if self._rank_mode == "downloads" else "Trending"
            self._set_model_status(f"[dim]{mode_label} models loaded. Select one to prefill repo-id.[/dim]")
            self._focus_model_list_if_pending()
        else:
            self._set_model_status("[yellow]No matching GGUF text-generation models found.[/yellow]")
            self._repo_to_quants = {}

    def on_storage_loaded(self, message: StorageLoaded) -> None:
        self._cached_models = _cached_models_from_snapshot(message.snapshot, BackendType.LLAMACPP)
        self._cached_repo_to_quants = {model.repo_id.casefold(): model.quantizations for model in self._cached_models}
        self._has_cached_snapshot = True
        if self._rank_mode == "cached":
            self._show_cached_models()

    def on_storage_failed(self, message: StorageFailed) -> None:
        if self._rank_mode != "cached":
            return
        self._ranked_models = []
        _set_option_list(self.query_one("#llama-model-list", OptionList), [])
        self._set_model_status(f"[yellow]Could not load cached models:[/yellow] {escape(message.error)}")

    def on_llama_cpp_models_failed(self, message: LlamaCppModelsFailed) -> None:
        if message.mode != self._rank_mode:
            return
        self._ranked_models = []
        self._repo_to_quants = {}
        _set_option_list(self.query_one("#llama-model-list", OptionList), [])
        self._set_model_status(
            f"[yellow]Could not load model suggestions:[/yellow] {escape(message.error)} [dim](manual input still works)[/dim]"
        )

    def on_llama_cpp_quants_loaded(self, message: LlamaCppQuantsLoaded) -> None:
        current_repo = self.query_one("#repo-id", Input).value.strip()
        current_revision = self.query_one("#revision", Input).value.strip()
        if message.repo_id.strip().casefold() != current_repo.casefold():
            return
        if (message.revision or "").strip() != current_revision:
            return
        repo_key = current_repo.casefold()
        self._repo_to_quants[repo_key] = tuple(message.quantizations)
        self._repo_to_quant_vram[repo_key] = _normalize_vram_map(message.vram_gb_by_quant)
        metadata_key = (repo_key, (message.revision or "").strip())
        self._repo_to_architecture[metadata_key] = message.architecture
        self._repo_to_compatibility[metadata_key] = (
            message.compatibility_status,
            message.compatibility_message,
            message.llamacpp_runtime_id,
        )
        self._apply_quantizations(
            self._display_quantizations_for_repo(repo_key, list(message.quantizations)),
            auto_select=not self._quant_touched,
            vram_gb_by_quant=self._repo_to_quant_vram[repo_key],
        )
        architecture_label = escape(message.architecture or "unknown")
        if message.compatibility_status == "supported":
            self._set_quant_status(
                "[dim]Quantizations:[/dim] "
                f"[green]Compatible architecture={architecture_label}[/green]"
            )
        elif message.compatibility_status == "unsupported":
            self._set_quant_status(
                "[dim]Quantizations:[/dim] "
                f"[red]Unsupported architecture {architecture_label}:[/red] "
                f"{escape(message.compatibility_message)}"
            )
        else:
            detail = (
                message.compatibility_message
                or "GGUF architecture metadata was not available."
            )
            self._set_quant_status(
                f"[dim]Quantizations:[/dim] "
                f"[yellow]Compatibility unknown:[/yellow] {escape(detail)}"
            )
        if self._prime_offers:
            self._refresh_prime_offer_options()

    def on_llama_cpp_quants_failed(self, message: LlamaCppQuantsFailed) -> None:
        current_repo = self.query_one("#repo-id", Input).value.strip()
        current_revision = self.query_one("#revision", Input).value.strip()
        if message.repo_id.strip().casefold() != current_repo.casefold():
            return
        if (message.revision or "").strip() != current_revision:
            return
        _set_option_list(self.query_one("#llama-quant-list", OptionList), [])
        self._set_quant_status(
            f"[yellow]Could not load quantizations:[/yellow] {escape(message.error)} [dim](manual quant still works)[/dim]"
        )

    def action_do_deploy(self) -> None:
        self._do_deploy()

    def action_predownload_highlighted(self) -> None:
        selected = self._highlighted_ranked_model()
        if selected is None:
            self.app.notify("Highlight a model in Model ranking first.", severity="warning", timeout=5)
            return
        quant = self.query_one("#quant", Input).value.strip() or None
        if quant is None and selected.quantizations:
            quant = selected.quantizations[0]
        revision = self.query_one("#revision", Input).value.strip() or None
        self.app.begin_storage_predownload(  # type: ignore[attr-defined]
            backend=BackendType.LLAMACPP,
            model_id=selected.repo_id,
            quant=quant,
            revision=revision,
        )

    def _do_deploy(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=self._provider,
        )
        config.repo_id = self.query_one("#repo-id", Input).value.strip() or None
        if not config.repo_id:
            self.app.notify(
                "Enter a Hugging Face repo-id before deploying.",
                severity="error",
                timeout=5,
            )
            self.query_one("#repo-id", Input).focus()
            return
        config.quant = self.query_one("#quant", Input).value.strip() or None
        config.required_vram_gb = self._current_llamacpp_required_vram()
        rev = self.query_one("#revision", Input).value.strip()
        config.revision = rev or None
        repo_key = (config.repo_id or "").casefold()
        metadata_key = (repo_key, rev)
        config.gguf_architecture = self._repo_to_architecture.get(metadata_key)
        compatibility = self._repo_to_compatibility.get(metadata_key)
        if compatibility is not None:
            status, message, runtime_id = compatibility
            config.llamacpp_runtime_id = runtime_id
            if status == "unsupported":
                self.app.notify(message, severity="error", timeout=8)
                return

        config.preload = False
        config.do_deploy = True
        config.do_warmup = self.query_one("#warmup", Switch).value
        config.show_debug_logs = self.query_one("#show-debug-logs-llama", Switch).value

        gpu_type = normalize_gpu_type(self._selected_gpu_type)
        if not gpu_type:
            self.app.notify("GPU type is required.", severity="error", timeout=5)
            return
        gpu_count = parse_gpu_count(self.query_one("#gpu-count-llama", Input).value, default=0)
        if gpu_count <= 0:
            self.app.notify("GPU count must be an integer >= 1.", severity="error", timeout=5)
            return
        config.gpu_type = gpu_type
        config.gpu_count = gpu_count
        if self._provider == ComputeProvider.PRIME:
            offer = self._prime_offers.get(self._selected_prime_offer_id or "")
            if offer is None:
                self.app.notify("Select a live Prime GPU offer.", severity="error", timeout=5)
                return
            if config.revision:
                self.app.notify(
                    "Prime llama.cpp currently supports only the default HF revision.",
                    severity="error",
                    timeout=5,
                )
                return
            config.gpu_type = offer.gpu_type
            config.gpu_count = offer.gpu_count
            allow_insecure_http = self.query_one(
                "#prime-insecure-http-llama", Switch
            ).value
            config.provider_options = PrimeProviderOptions(
                offer_id=offer.id,
                disk_id=self.query_one("#prime-disk-id-llama", Input).value.strip() or None,
                allow_insecure_http=allow_insecure_http,
                keep_failed_resource=self.query_one(
                    "#prime-keep-failed-llama", Switch
                ).value,
                auto_disk=self.query_one("#prime-auto-disk-llama", Switch).value,
            )
        elif self._provider == ComputeProvider.VAST:
            vast_options = self._build_vast_provider_options("#vast-disk-llama")
            if vast_options is None:
                return
            config.provider_options = vast_options
            vast_offer = self._vast_offers[vast_options.offer_id]
            config.gpu_type = vast_offer.gpu_type
            config.gpu_count = vast_offer.gpu_count

        # Advanced values are always read: collapsing the section must never
        # silently discard options the user entered before collapsing it.
        sa = self.query_one("#server-args", Input).value.strip()
        config.server_args = sa or None
        h = self.query_one("#host-input", Input).value.strip()
        config.host = h or None
        p = self.query_one("#port-input", Input).value.strip()
        if p:
            config.port = positive_int(p)
            if config.port is None or config.port > 65535:
                self.app.notify("Port must be an integer from 1 to 65535.", severity="error", timeout=5)
                return
        ngl = self.query_one("#n-gpu-layers", Input).value.strip()
        if ngl:
            try:
                config.n_gpu_layers = int(ngl)
            except ValueError:
                self.app.notify("GPU layers must be an integer, or blank for auto.", severity="error", timeout=5)
                return
        config.llamacpp_image_no_cache = self.query_one("#llama-image-no-cache", Switch).value

        model_hint = config.repo_id
        instance_override = self.query_one("#instance-name-llama", Input).value.strip()
        app_override = self.query_one("#app-name-llama", Input).value.strip()
        if app_override:
            config.app_name = app_override
            config.instance_name = slugify_instance_name(instance_override or app_override)
        elif instance_override:
            config.instance_name = slugify_instance_name(instance_override)
            config.app_name = build_deployment_name(
                config.provider, config.backend, config.instance_name
            )
        else:
            config.instance_name = auto_instance_name_for_backend(config.backend, model_hint)
            config.app_name = build_deployment_name(
                config.provider, config.backend, config.instance_name
            )

        try:
            self.query_one(VisionOptions).apply(config)
        except ValueError as exc:
            self.app.notify(str(exc), severity="error", timeout=8)
            return
        # Refuse here rather than after routing, so the form never accepts a
        # configuration its provider will reject.
        reason = refuse_deployment(config)
        if reason:
            self.app.notify(reason, severity="error", timeout=8)
            return
        self.app.begin_deploy(config)  # type: ignore[attr-defined]

    def _focus_model_list_if_pending(self) -> None:
        """Give the model picker focus once a requested ranking has loaded."""
        if not self._focus_model_list_when_loaded:
            return
        model_list = self.query_one(f"#{self._MODEL_LIST_ID}", OptionList)
        if model_list.option_count == 0:
            return
        self._focus_model_list_when_loaded = False
        model_list.focus()
        if model_list.highlighted is None:
            model_list.highlighted = 0

    def _set_model_status(self, text: str) -> None:
        self.query_one("#llama-model-status", Static).update(text)

    def _set_ranking_title(self) -> None:
        subtitle = _ranking_subtitle(BackendType.LLAMACPP, self._rank_mode)
        try:
            self.query_one("#llama-model-ranking-title", Static).update(
                f"[bold]Model[/bold]  [dim]({subtitle})[/dim]"
            )
        except Exception:
            return

    def _set_quant_status(self, text: str) -> None:
        self.query_one("#llama-quant-status", Static).update(text)

    def _resolve_rank_mode(self, option_id: str) -> str | None:
        if option_id == "rank-cached":
            return "cached"
        if option_id == "rank-downloads":
            return "downloads"
        if option_id == "rank-trending":
            return "trending"
        return None

    def _refresh_cached_models_from_storage(self, force: bool = False) -> None:
        refresher = getattr(self.app, "begin_storage_refresh", None)
        if callable(refresher):
            refresher(self, force=force)

    def _show_cached_models(self) -> None:
        self._ranked_models = list(self._cached_models)
        self._repo_to_quants = {model.repo_id.casefold(): model.quantizations for model in self._ranked_models}
        model_list = self.query_one("#llama-model-list", OptionList)
        options = []
        for idx, model in enumerate(self._ranked_models):
            quant_preview = ", ".join(model.quantizations[:3]) if model.quantizations else "-"
            if len(model.quantizations) > 3:
                quant_preview = f"{quant_preview}, ..."
            label = _model_row(model.repo_id, f"quants={quant_preview}")
            options.append(Option(label, id=f"model-{idx}"))
        _set_option_list(model_list, options)
        if self._ranked_models:
            self._set_model_status("[dim]Cached models loaded. Select one to prefill repo-id.[/dim]")
            self._focus_model_list_if_pending()
        elif self._fall_back_to_downloads_ranking():
            return
        else:
            self._set_model_status("[yellow]No cached llama.cpp models found in storage.[/yellow]")

    def _fall_back_to_downloads_ranking(self) -> bool:
        """Switch an empty default "Cached in storage" view to "Most downloaded".

        A first-time user has nothing cached, so the default ranking mode opens
        the form on an empty picker and a warning. Only the automatic default
        is replaced: once the mode has been chosen by hand it is left alone.
        """
        if self._rank_mode_touched or self._rank_mode != "cached" or not self._has_cached_snapshot:
            return False
        self._rank_mode = "downloads"
        rank_mode_list = self.query_one("#llama-rank-mode", OptionList)
        rank_mode_list.highlighted = 1
        self._set_ranking_title()
        self._set_model_status("[dim]Nothing cached yet. Loading popular models...[/dim]")
        self.app.begin_fetch_llamacpp_models("downloads", self)  # type: ignore[attr-defined]
        return True

    def _apply_ranked_model_selection(self, option_id: str) -> None:
        selected = _model_from_option_id(option_id, self._ranked_models)
        if selected is None:
            return
        self.query_one("#repo-id", Input).value = selected.repo_id
        self._quant_touched = False
        repo_key = selected.repo_id.casefold()
        cached_vram = self._repo_to_quant_vram.get(repo_key, {})
        if selected.quantizations:
            self._apply_quantizations(
                self._display_quantizations_for_repo(repo_key, list(selected.quantizations)),
                auto_select=True,
                vram_gb_by_quant=cached_vram,
            )
            if not cached_vram:
                self._lookup_quantizations_for_current_repo()
        else:
            self._lookup_quantizations_for_current_repo(force_refresh=True)
        self._refresh_app_preview()

    def _highlighted_ranked_model(self) -> ModelCandidate | None:
        highlighted = self.query_one("#llama-model-list", OptionList).highlighted_option
        option_id = highlighted.id if highlighted is not None else ""
        return _model_from_option_id(option_id or "", self._ranked_models)

    def _display_quantizations_for_repo(self, repo_key: str, quantizations: list[str]) -> list[str]:
        """Return every known quantization for a repo, cached or not.

        Which quantizations a repo publishes has nothing to do with which model
        list is being browsed, so the ranking mode must not narrow this. What
        is already in storage is shown as a per-row marker instead.
        """
        cached_quants = self._cached_repo_to_quants.get(repo_key) or ()
        merged = list(quantizations)
        seen = {quant.strip().upper() for quant in merged}
        merged.extend(quant for quant in cached_quants if quant.strip().upper() not in seen)
        return merged

    def _cached_quants_for_current_repo(self) -> set[str]:
        repo_key = self.query_one("#repo-id", Input).value.strip().casefold()
        return {quant.strip().upper() for quant in self._cached_repo_to_quants.get(repo_key, ())}

    def _lookup_quantizations_for_current_repo(self, force_refresh: bool = False) -> None:
        self._cancel_quantization_lookup()
        repo_id = self.query_one("#repo-id", Input).value.strip()
        revision = self.query_one("#revision", Input).value.strip()
        if not repo_id:
            _set_option_list(self.query_one("#llama-quant-list", OptionList), [])
            self._set_quant_status("[dim]Quantizations: enter repo-id to detect GGUF variants[/dim]")
            self._last_quant_lookup = None
            return

        repo_key = repo_id.casefold()
        cache_key = (repo_key, revision)
        cached_quants = self._repo_to_quants.get(repo_key)
        cached_vram = self._repo_to_quant_vram.get(repo_key, {})
        if cached_quants is not None and not force_refresh:
            self._apply_quantizations(
                self._display_quantizations_for_repo(repo_key, list(cached_quants)),
                auto_select=not self._quant_touched,
                vram_gb_by_quant=cached_vram,
            )
            if cached_vram or not cached_quants:
                return

        if not force_refresh and cache_key == self._last_quant_lookup:
            return
        self._last_quant_lookup = cache_key

        self._set_quant_status("[dim]Loading quantizations...[/dim]")
        self.app.begin_fetch_llamacpp_quants(repo_id, revision or None, self)  # type: ignore[attr-defined]

    def _cancel_quantization_lookup(self) -> None:
        timer = self._quant_lookup_timer
        self._quant_lookup_timer = None
        if timer is not None:
            timer.stop()

    def _schedule_quantization_lookup(self) -> None:
        """Debounce remote GGUF metadata lookups while the user is typing."""
        self._cancel_quantization_lookup()
        repo_id = self.query_one("#repo-id", Input).value.strip()
        if not repo_id:
            self._lookup_quantizations_for_current_repo()
            return
        if not _is_plausible_model_lookup(repo_id):
            self._last_quant_lookup = None
            self._set_quant_status("[dim]Quantizations: finish entering the repo-id[/dim]")
            return
        self._quant_lookup_timer = self.set_timer(
            _MODEL_LOOKUP_DEBOUNCE_SECONDS,
            self._run_scheduled_quantization_lookup,
            name="llamacpp-quantization-lookup-debounce",
        )

    def _run_scheduled_quantization_lookup(self) -> None:
        self._quant_lookup_timer = None
        self._lookup_quantizations_for_current_repo()

    def _apply_quantizations(
        self,
        quantizations: list[str],
        auto_select: bool,
        vram_gb_by_quant: dict[str, float] | None = None,
    ) -> None:
        normalized_vram = _normalize_vram_map(vram_gb_by_quant)
        sorted_quantizations: list[str] = []
        if normalized_vram:
            sorted_quantizations = sorted(
                quantizations,
                key=lambda q: normalized_vram.get(q.strip().upper(), float("inf"))
            )
        else:
            sorted_quantizations = list(quantizations)
        quant_list = self.query_one("#llama-quant-list", OptionList)
        cached = self._cached_quants_for_current_repo()
        options = []
        for quant in sorted_quantizations:
            label = f"  {_format_quant_with_vram(quant, normalized_vram)}"
            if quant.strip().upper() in cached:
                label = f"{label}  [dim]· in storage[/dim]"
            options.append(Option(label, id=f"quant-{quant}"))
        _set_option_list(quant_list, options)
        if quantizations:
            self._set_quant_status("[dim]Quantizations:[/dim]")
        else:
            self._set_quant_status("[yellow]No GGUF quantizations detected.[/yellow]")

        quant_input = self.query_one("#quant", Input)
        current_quant = quant_input.value.strip().upper()
        if not auto_select:
            return
        if current_quant and current_quant in {q.upper() for q in quantizations}:
            return
        preferred = "Q4_K_M" if "Q4_K_M" in quantizations else (quantizations[0] if quantizations else "")
        if preferred:
            self._updating_quant_input = True
            try:
                quant_input.value = preferred
            finally:
                self._updating_quant_input = False

    def _apply_quant_selection(self, option_id: str) -> None:
        if not option_id.startswith("quant-"):
            return
        quant = option_id.removeprefix("quant-").strip()
        if not quant:
            return
        self._updating_quant_input = True
        try:
            self.query_one("#quant", Input).value = quant
        finally:
            self._updating_quant_input = False
        self._quant_touched = True

    def _refresh_app_preview(self) -> None:
        repo_id = self.query_one("#repo-id", Input).value.strip()
        model_hint = repo_id or "default"
        instance_override = self.query_one("#instance-name-llama", Input).value.strip()
        app_override = self.query_one("#app-name-llama", Input).value.strip()
        if app_override:
            preview = app_override
        elif instance_override:
            preview = build_deployment_name(
                self._provider,
                BackendType.LLAMACPP,
                instance_override,
            )
        else:
            preview = build_deployment_name(
                self._provider,
                BackendType.LLAMACPP,
                auto_instance_name_for_backend(BackendType.LLAMACPP, model_hint),
            )
        self.query_one("#llama-app-preview", Static).update(f"[dim]App name preview: {escape(preview)}[/dim]")

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_open_storage(self) -> None:
        self.app.action_push_storage(BackendType.LLAMACPP)  # type: ignore[attr-defined]


class VllmDeployScreen(_OptionListArrowNavigationMixin, _CostPreviewMixin, CopyEnabledScreen):
    """vLLM deploy form."""

    BINDINGS = [
        Binding("up", "navigate_option_list_up", show=False, priority=True),
        Binding("down", "navigate_option_list_down", show=False, priority=True),
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+d", "do_deploy", "Deploy", show=True, priority=True),
        Binding("ctrl+s", "open_storage", "Storage", show=True),
        Binding("p", "predownload_highlighted", "Pre-download", show=True),
    ]
    _MODEL_LIST_ID = "vllm-model-list"
    OPTION_LIST_IDS = ("vllm-rank-mode", "vllm-model-list")
    NAVIGATION_ORDER = (
        "vllm-rank-mode",
        "vllm-model-list",
        "model-name",
        "provider-vllm",
        "prime-offer-vllm",
        "vast-offer-vllm",
        "gpu-type-vllm",
        "gpu-count-vllm",
        "n-gpu",
        "vision-mode",
        "image-limit",
        "mm-processor-kwargs",
        "toggle-advanced-vllm",
        "model-revision",
        "prime-auto-disk",
        "prime-disk-id",
        "prime-insecure-http",
        "prime-keep-failed",
        "vast-disk-vllm",
        "smoke-only-vllm",
        "warmup-vllm",
        "fast-boot",
        "trust-remote-code",
        "show-debug-logs-vllm",
        "served-model-name",
        "reasoning-parser",
        "tool-call-parser",
        "chat-template-kwargs",
        "instance-name-vllm",
        "app-name-vllm",
        "deploy-vllm-btn",
    )

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="screen-scroll"):
            yield Static("[bold #7bf168]Advanced deploy vLLM[/]  [dim]Step 2: Model & options[/dim]")
            yield Static("")

            yield Static(
                "[bold]Model[/bold]  [dim](cached models in your storage volumes)[/dim]",
                id="vllm-model-ranking-title",
            )
            yield OptionList(
                Option("  Cached in storage", id="rank-cached"),
                Option("  Most downloaded", id="rank-downloads"),
                Option("  Trending", id="rank-trending"),
                id="vllm-rank-mode",
            )
            yield Static("[dim]Loading model suggestions...[/dim]", id="vllm-model-status")
            yield OptionList(id="vllm-model-list")
            yield Static("")

            yield FormField(
                "Model name",
                "model-name",
            )
            yield Static("[dim]Estimated VRAM: enter model name to compute[/dim]", id="vllm-vram-status")
            yield Static("Compute provider", classes="form-label")
            yield Select(
                options=[
                    ("Modal", "modal"),
                    ("Prime Intellect", "prime"),
                    ("Vast.ai", "vast"),
                ],
                value="modal",
                allow_blank=False,
                id="provider-vllm",
            )
            yield Static("Prime GPU offer", classes="form-label prime-only")
            yield Select(
                options=[],
                prompt="Select an exact live Prime offer",
                id="prime-offer-vllm",
                classes="prime-only",
            )
            yield Static(
                "[dim]Prime offers are secure, on-demand availability sorted by price.[/dim]",
                id="prime-offer-status",
                classes="prime-only",
            )
            yield Static("Vast.ai rental", classes="form-label vast-only")
            yield Select(
                options=[],
                prompt="Select a live Vast.ai rental",
                id="vast-offer-vllm",
                classes="vast-only",
            )
            yield Static(
                "[dim]Vast.ai rentals are priced including disk.[/dim]",
                id="vast-offer-status-vllm",
                classes="vast-only",
            )
            with Vertical(classes="gpu-config-panel"):
                yield Static("GPU configuration", classes="form-section-title")
                yield Static(
                    "Choose deployment GPUs and in-replica tensor sharding. "
                    "Base Modal hourly price per GPU is shown when available.",
                    id="gpu-config-subtitle-vllm",
                    classes="form-section-subtitle",
                )
                with Horizontal(id="gpu-config-row-vllm", classes="gpu-config-main-row"):
                    with Vertical(id="gpu-type-group-vllm"):
                        yield Static("GPU type", classes="form-label")
                        yield Select(
                            options=[(DEFAULT_GPU_TYPE, DEFAULT_GPU_TYPE)],
                            prompt="Select GPU type",
                            value=DEFAULT_GPU_TYPE,
                            id="gpu-type-vllm",
                        )
                    with Vertical(id="gpu-count-group-vllm"):
                        yield Static("GPUs attached", classes="form-label")
                        yield Input(
                            value=str(DEFAULT_GPU_COUNT),
                            placeholder="1",
                            id="gpu-count-vllm",
                            type="integer",
                        )
                yield FormField(
                    "Tensor parallel size",
                    "n-gpu",
                    default="1",
                    input_type="integer",
                    hint="Shards one model across the attached GPUs "
                    "(vLLM --tensor-parallel-size)",
                    classes="gpu-config-tensor-field",
                )
                yield Static("", id="vllm-cost-preview")
            yield VisionOptions(BackendType.VLLM)
            yield Button("Advanced options...", id="toggle-advanced-vllm", variant="default")
            yield FormField(
                "Model revision (optional)",
                "model-revision",
                hint="Leave blank to use default branch",
                classes="vllm-advanced",
            )
            yield ToggleField(
                "Attach persistent cache disk",
                "prime-auto-disk",
                default=True,
                classes="vllm-advanced prime-only",
            )
            yield FormField(
                "Prime disk ID (optional)",
                "prime-disk-id",
                hint="Leave blank to auto-attach a persistent cache disk",
                classes="vllm-advanced prime-only",
            )
            yield ToggleField(
                "Use direct HTTP fallback (insecure)",
                "prime-insecure-http",
                default=False,
                classes="vllm-advanced prime-only",
            )
            yield ToggleField(
                "Keep failed Prime pod (billing may continue)",
                "prime-keep-failed",
                default=False,
                classes="vllm-advanced prime-only",
            )
            yield FormField(
                "Vast disk size (GB)",
                "vast-disk-vllm",
                default=str(DEFAULT_VAST_DISK_GB),
                input_type="integer",
                hint="Rented alongside the GPU and included in the quoted price",
                classes="vllm-advanced vast-only",
            )
            yield ToggleField(
                "Smoke test only (no deploy)",
                "smoke-only-vllm",
                default=False,
                classes="vllm-advanced",
            )
            yield ToggleField(
                "Warm up after deploy",
                "warmup-vllm",
                default=True,
                classes="vllm-advanced",
            )
            yield ToggleField(
                "Enforce eager startup (skips CUDA graph capture)",
                "fast-boot",
                default=False,
                classes="vllm-advanced",
            )
            yield ToggleField(
                "Trust remote model code",
                "trust-remote-code",
                default=False,
                classes="vllm-advanced",
            )
            yield ToggleField(
                "Show debug logs (full raw backend logs)",
                "show-debug-logs-vllm",
                default=False,
                classes="vllm-advanced",
            )
            yield Static("Runtime", classes="form-group-heading vllm-advanced")
            yield FormField(
                "Served model alias",
                "served-model-name",
                hint="Defaults to the model id suffix (e.g., Qwen3-0.6B)",
                classes="vllm-advanced",
            )
            yield FormField(
                "Reasoning parser (optional)",
                "reasoning-parser",
                hint="e.g., qwen3, deepseek_r1, granite",
                classes="vllm-advanced",
            )
            yield FormField(
                "Tool call parser (optional)",
                "tool-call-parser",
                hint="e.g., hermes, qwen3_xml, llama3_json",
                classes="vllm-advanced",
            )
            yield FormField(
                "Default chat template kwargs (JSON, optional)",
                "chat-template-kwargs",
                hint='e.g., {"enable_thinking": false}',
                classes="vllm-advanced",
            )
            yield Static("Naming", classes="form-group-heading vllm-advanced")
            yield FormField(
                "Instance name (optional)",
                "instance-name-vllm",
                hint="Auto-derived from model if blank",
                classes="vllm-advanced",
            )
            yield FormField(
                "App name override (optional)",
                "app-name-vllm",
                hint="Advanced: explicit deployment name",
                classes="vllm-advanced",
            )
            yield Static(
                "[dim]App name preview: auto[/dim]",
                id="vllm-app-preview",
                classes="vllm-advanced",
            )

            yield Static("")
            yield Button("Deploy", id="deploy-vllm-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self._rank_mode = "cached"
        self._ranked_models: list[ModelCandidate] = []
        self._cached_models: list[ModelCandidate] = []
        self._has_cached_snapshot = False
        self._selected_gpu_type = DEFAULT_GPU_TYPE
        # Binding a Prime offer overwrites the GPU dropdown with that offer's
        # GPU. Remember the Modal choice so switching back does not leave a
        # Prime-only GPU name selected for a Modal deploy.
        self._last_modal_gpu_type = DEFAULT_GPU_TYPE
        self._modal_gpu_values: set[str] = {DEFAULT_GPU_TYPE}
        self._provider = ComputeProvider.MODAL
        self._prime_offers: dict[str, ComputeOffer] = {}
        self._selected_prime_offer_id: str | None = None
        self._vast_offers: dict[str, VastOffer] = {}
        self._selected_vast_offer_id: str | None = None
        self._served_alias_touched = False
        self._updating_served_alias = False
        self._last_auto_served_alias = ""
        self._tool_call_parser_touched = False
        self._updating_tool_call_parser = False
        self._last_auto_tool_call_parser = ""
        self._model_to_memory_estimate: dict[str, VllmMemoryBreakdown | None] = {}
        self._last_memory_lookup: tuple[str, str] | None = None
        self._memory_lookup_timer: Timer | None = None
        self._gpu_price_by_value: dict[str, float] = {}
        self._rank_mode_touched = False
        self._focus_model_list_when_loaded = False
        self.query_one("#vllm-model-list", OptionList).add_class("hidden")
        for widget in self.query(".vllm-advanced"):
            widget.add_class("hidden")
        for widget in self.query(".prime-only"):
            widget.add_class("hidden")
        for widget in self.query(".vast-only"):
            widget.add_class("hidden")
        rank_mode_list = self.query_one("#vllm-rank-mode", OptionList)
        if rank_mode_list.option_count > 0:
            rank_mode_list.highlighted = 0
        rank_mode_list.focus()
        self._set_ranking_title()
        self._refresh_gpu_types()
        self._set_model_status("[dim]Loading cached models from storage...[/dim]")
        self._refresh_cached_models_from_storage()
        self._sync_served_alias_from_model(force=True)
        self._sync_tool_call_parser_from_model(force=True)
        self._refresh_app_preview()
        self._update_cost_preview("vllm-cost-preview")
        self._refresh_vllm_memory_status()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "vllm-rank-mode":
            selected_mode = self._resolve_rank_mode(event.option.id or "")
            if selected_mode is None:
                return
            self._rank_mode_touched = True
            if selected_mode != self._rank_mode:
                self._rank_mode = selected_mode
                self._set_ranking_title()
                self._ranked_models = []
                self._set_model_status("[dim]Loading model suggestions...[/dim]")
                _set_option_list(self.query_one("#vllm-model-list", OptionList), [])
                if self._rank_mode == "cached":
                    self._set_model_status("[dim]Loading cached models from storage...[/dim]")
                    if self._has_cached_snapshot:
                        self._show_cached_models()
                    self._refresh_cached_models_from_storage()
                else:
                    self.app.begin_fetch_vllm_models(self._rank_mode, self)  # type: ignore[attr-defined]
            model_list = self.query_one("#vllm-model-list", OptionList)
            # The picker is hidden until it has rows, so a still-loading list
            # cannot take focus yet. Hand it over once the rows land.
            self._focus_model_list_when_loaded = model_list.option_count == 0
            _advance_deploy_focus(self, self.NAVIGATION_ORDER)
            return

        if event.option_list.id == "vllm-model-list":
            self._apply_ranked_model_selection(event.option.id or "")
            self._refresh_app_preview()
            _advance_deploy_focus(self, self.NAVIGATION_ORDER)
            return

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Mirror highlighted model into the input for keyboard-first flow."""
        if event.option_list.id != "vllm-model-list":
            return
        self._apply_ranked_model_selection(event.option.id or "")
        self._refresh_app_preview()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "toggle-advanced-vllm":
            for widget in self.query(".vllm-advanced"):
                widget.toggle_class("hidden")
            self._sync_prime_visibility()
        elif event.button.id == "deploy-vllm-btn":
            self._do_deploy()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-name":
            self._sync_served_alias_from_model()
            self._sync_tool_call_parser_from_model()
            self._schedule_vllm_memory_refresh()
        elif event.input.id == "model-revision":
            self._schedule_vllm_memory_refresh()
        elif event.input.id == "n-gpu":
            self._refresh_vllm_memory_status(from_cache_only=True)
        elif event.input.id == "served-model-name" and not self._updating_served_alias:
            self._served_alias_touched = event.input.value.strip() != self._last_auto_served_alias
        elif event.input.id == "tool-call-parser" and not self._updating_tool_call_parser:
            self._tool_call_parser_touched = (
                event.input.value.strip() != self._last_auto_tool_call_parser
            )
        if event.input.id in {"model-name", "instance-name-vllm", "app-name-vllm"}:
            self._refresh_app_preview()
        if event.input.id in {"model-name", "model-revision"} and self._prime_offers:
            self._refresh_prime_offer_options()
        if event.input.id == "gpu-count-vllm":
            self._update_cost_preview("vllm-cost-preview")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "provider-vllm":
            if not isinstance(event.value, str):
                return
            self._provider = ComputeProvider(event.value)
            self._sync_prime_visibility()
            if self._provider == ComputeProvider.PRIME and not self._prime_offers:
                self._refresh_prime_offers()
            elif self._provider == ComputeProvider.VAST:
                if self._vast_offers:
                    self._refresh_vast_offer_options()
                else:
                    self._refresh_vast_offers()
            elif self._provider == ComputeProvider.MODAL:
                self._selected_gpu_type = self._last_modal_gpu_type
                self._selected_prime_offer_id = None
                self._selected_vast_offer_id = None
                self._refresh_gpu_types()
            self._refresh_app_preview()
            self._update_cost_preview("vllm-cost-preview")
            return
        if event.select.id == "vast-offer-vllm":
            if not isinstance(event.value, str):
                return
            self._selected_vast_offer_id = event.value
            offer = self._vast_offers.get(event.value)
            if offer is not None:
                self._selected_gpu_type = offer.gpu_type
                gpu_type_select = self.query_one("#gpu-type-vllm", Select)
                gpu_type_select.set_options([(offer.gpu_type, offer.gpu_type)])
                gpu_type_select.value = offer.gpu_type
                self.query_one("#gpu-count-vllm", Input).value = str(offer.gpu_count)
                self.query_one("#n-gpu", Input).value = str(offer.gpu_count)
            self._update_cost_preview("vllm-cost-preview")
            return
        if event.select.id == "prime-offer-vllm":
            if not isinstance(event.value, str):
                return
            self._selected_prime_offer_id = event.value
            offer = self._prime_offers.get(event.value)
            if offer is not None:
                self._selected_gpu_type = offer.gpu_type
                self.query_one("#gpu-type-vllm", Select).set_options(
                    [(offer.gpu_type, offer.gpu_type)]
                )
                self.query_one("#gpu-type-vllm", Select).value = offer.gpu_type
                self.query_one("#gpu-count-vllm", Input).value = str(offer.gpu_count)
                self.query_one("#n-gpu", Input).value = str(offer.gpu_count)
            self._update_cost_preview("vllm-cost-preview")
            return
        if event.select.id != "gpu-type-vllm":
            return
        if isinstance(event.value, str) and event.value.strip():
            self._selected_gpu_type = normalize_gpu_type(event.value)
            if self._selected_gpu_type in self._modal_gpu_values:
                self._last_modal_gpu_type = self._selected_gpu_type
        self._update_cost_preview("vllm-cost-preview")

    def _sync_prime_visibility(self) -> None:
        self.query_one("#gpu-config-subtitle-vllm", Static).update(
            _gpu_panel_subtitle(BackendType.VLLM, self._provider)
        )
        # A bound Prime offer dictates the GPU shape and count, and the deploy
        # path overwrites whatever these hold. Leaving them editable invites
        # changes that are silently discarded.
        offer_bound = self._provider in (ComputeProvider.PRIME, ComputeProvider.VAST)
        for field_id in ("#gpu-type-vllm", "#gpu-count-vllm", "#n-gpu",):
            self.query_one(field_id).disabled = offer_bound
        advanced_visible = not self.query_one("#model-revision").has_class("hidden")
        for provider_class, provider in (
            (".prime-only", ComputeProvider.PRIME),
            (".vast-only", ComputeProvider.VAST),
        ):
            for widget in self.query(provider_class):
                hide = self._provider != provider
                if widget.has_class("vllm-advanced") and not advanced_visible:
                    hide = True
                widget.set_class(hide, "hidden")

    def _refresh_prime_offers(self) -> None:
        self.query_one("#prime-offer-status", Static).update("[dim]Loading Prime offers...[/dim]")
        self.run_worker(
            self._run_fetch_prime_offers,
            name="vllm-fetch-prime-offers",
            thread=True,
        )

    def _run_fetch_prime_offers(self) -> None:
        try:
            offers = PrimeBackend().list_offers()
        except Exception as exc:
            self.post_message(PrimeOffersFailed(str(exc)))
            return
        self.post_message(PrimeOffersLoaded(offers))

    def on_prime_offers_loaded(self, message: PrimeOffersLoaded) -> None:
        self._prime_offers = {offer.id: offer for offer in message.offers}
        self._refresh_prime_offer_options()

    def _current_vllm_required_vram(self) -> float | None:
        repo_id, revision = self._current_memory_lookup()
        if not repo_id:
            return None
        estimate = self._model_to_memory_estimate.get(
            self._memory_cache_key(repo_id, revision)
        )
        return estimate.total_gb if estimate is not None else None

    def _refresh_prime_offer_options(self) -> None:
        required_vram_gb = self._current_vllm_required_vram()
        offers = _compatible_prime_offers(
            list(self._prime_offers.values()),
            required_vram_gb,
            BackendType.VLLM,
        )
        options = _prime_offer_options(offers)
        selector = self.query_one("#prime-offer-vllm", Select)
        selector.set_options(options)
        if options:
            self._selected_prime_offer_id = options[0][1]
            selector.value = options[0][1]
        else:
            self._selected_prime_offer_id = None
        self.query_one("#prime-offer-status", Static).update(
            _prime_offer_status(
                len(options),
                required_vram_gb,
                BackendType.VLLM,
            )
        )

    def on_prime_offers_failed(self, message: PrimeOffersFailed) -> None:
        self.query_one("#prime-offer-status", Static).update(
            f"[red]Could not load Prime offers:[/red] {escape(message.error)}"
        )

    def _refresh_vast_offers(self) -> None:
        self.query_one("#vast-offer-status-vllm", Static).update(
            "[dim]Loading Vast.ai rentals...[/dim]"
        )
        self.run_worker(
            self._run_fetch_vast_offers,
            name="vllm-fetch-vast-offers",
            thread=True,
        )

    def _run_fetch_vast_offers(self) -> None:
        try:
            offers = VastBackend().list_offers(
                VastOfferQuery(gpu_count=None, disk_gb=self._vast_disk_gb())
            )
        except Exception as exc:
            self.post_message(VastOffersFailed(str(exc)))
            return
        self.post_message(VastOffersLoaded(offers))

    def on_vast_offers_loaded(self, message: VastOffersLoaded) -> None:
        self._vast_offers = {offer.id: offer for offer in message.offers}
        self._refresh_vast_offer_options()

    def on_vast_offers_failed(self, message: VastOffersFailed) -> None:
        self.query_one("#vast-offer-status-vllm", Static).update(
            f"[red]Could not load Vast.ai rentals:[/red] {escape(message.error)}"
        )

    def _vast_disk_gb(self) -> int:
        return positive_int(self.query_one("#vast-disk-vllm", Input).value) or DEFAULT_VAST_DISK_GB

    def _refresh_vast_offer_options(self) -> None:
        required_vram_gb = self._current_vllm_required_vram()
        offers = _compatible_vast_offers(list(self._vast_offers.values()), required_vram_gb)
        options = _vast_offer_options(offers)
        selector = self.query_one("#vast-offer-vllm", Select)
        selector.set_options(options)
        if options:
            self._selected_vast_offer_id = options[0][1]
            selector.value = options[0][1]
        else:
            self._selected_vast_offer_id = None
        self.query_one("#vast-offer-status-vllm", Static).update(
            _vast_offer_status(len(options), required_vram_gb)
        )

    def _refresh_gpu_types(self) -> None:
        self.run_worker(
            lambda: self._run_fetch_gpu_types(),
            name="vllm-fetch-gpu-types",
            thread=True,
        )

    def _run_fetch_gpu_types(self) -> None:
        poster = getattr(self, "post_message", None)
        if poster is None:
            return
        try:
            gpu_types = fetch_modal_gpu_catalog()
        except Exception as exc:
            poster(GpuTypesFailed(error=str(exc)))
            return
        poster(GpuTypesLoaded(gpu_types=gpu_types))

    def on_gpu_types_loaded(self, message: GpuTypesLoaded) -> None:
        dropdown = self.query_one("#gpu-type-vllm", Select)
        options = build_gpu_type_options(message.gpu_types)
        option_values = [value for _, value in options]
        self._modal_gpu_values = set(option_values)
        if self._selected_gpu_type and self._selected_gpu_type not in option_values:
            options.insert(0, (self._selected_gpu_type, self._selected_gpu_type))
            option_values.insert(0, self._selected_gpu_type)
        if not options:
            return
        dropdown.set_options(options)
        selected = self._selected_gpu_type if self._selected_gpu_type in option_values else option_values[0]
        dropdown.value = selected
        self._selected_gpu_type = selected
        self._gpu_price_by_value = {
            spec.value.strip(): spec.price_per_hour_usd
            for spec in message.gpu_types
            if spec.price_per_hour_usd is not None
        }
        self._update_cost_preview("vllm-cost-preview")

    def on_gpu_types_failed(self, _: GpuTypesFailed) -> None:
        # Keep default value if Modal docs fetch fails.
        return

    def _sync_served_alias_from_model(self, force: bool = False) -> None:
        alias_input = self.query_one("#served-model-name", Input)
        next_alias = default_served_model_name(self.query_one("#model-name", Input).value)
        current_alias = alias_input.value.strip()
        should_update = force or (not self._served_alias_touched) or (current_alias == self._last_auto_served_alias)
        if not should_update or current_alias == next_alias:
            self._last_auto_served_alias = next_alias
            return
        self._updating_served_alias = True
        try:
            alias_input.value = next_alias
        finally:
            self._updating_served_alias = False
        self._last_auto_served_alias = next_alias

    def _sync_tool_call_parser_from_model(self, force: bool = False) -> None:
        """Prefill known-safe tool calling without overwriting a user choice."""

        parser_input = self.query_one("#tool-call-parser", Input)
        next_parser = recommended_vllm_tool_call_parser(
            self.query_one("#model-name", Input).value
        ) or ""
        current_parser = parser_input.value.strip()
        should_update = (
            force
            or not self._tool_call_parser_touched
            or current_parser == self._last_auto_tool_call_parser
        )
        if not should_update or current_parser == next_parser:
            self._last_auto_tool_call_parser = next_parser
            return
        self._updating_tool_call_parser = True
        try:
            parser_input.value = next_parser
        finally:
            self._updating_tool_call_parser = False
        self._last_auto_tool_call_parser = next_parser

    def on_vllm_models_loaded(self, message: VllmModelsLoaded) -> None:
        if message.mode != self._rank_mode:
            return
        self._ranked_models = message.models
        model_list = self.query_one("#vllm-model-list", OptionList)
        options = []
        for idx, model in enumerate(message.models):
            downloads = f"{model.downloads:,}" if model.downloads is not None else "-"
            likes = f"{model.likes:,}" if model.likes is not None else "-"
            label = _model_row(model.repo_id, f"downloads={downloads:<10} likes={likes}")
            options.append(Option(label, id=f"model-{idx}"))
        _set_option_list(model_list, options)

        if message.models:
            mode_label = "Most downloaded" if self._rank_mode == "downloads" else "Trending"
            self._set_model_status(f"[dim]{mode_label} models loaded. Select one to prefill Model name.[/dim]")
            self._focus_model_list_if_pending()
        else:
            self._set_model_status("[yellow]No matching text-generation models found.[/yellow]")

    def on_storage_loaded(self, message: StorageLoaded) -> None:
        self._cached_models = _cached_models_from_snapshot(message.snapshot, BackendType.VLLM)
        self._has_cached_snapshot = True
        if self._rank_mode == "cached":
            self._show_cached_models()

    def on_storage_failed(self, message: StorageFailed) -> None:
        if self._rank_mode != "cached":
            return
        self._ranked_models = []
        _set_option_list(self.query_one("#vllm-model-list", OptionList), [])
        self._set_model_status(f"[yellow]Could not load cached models:[/yellow] {escape(message.error)}")

    def on_vllm_models_failed(self, message: VllmModelsFailed) -> None:
        if message.mode != self._rank_mode:
            return
        self._ranked_models = []
        _set_option_list(self.query_one("#vllm-model-list", OptionList), [])
        self._set_model_status(
            f"[yellow]Could not load model suggestions:[/yellow] {escape(message.error)} [dim](manual input still works)[/dim]"
        )

    def action_do_deploy(self) -> None:
        self._do_deploy()

    def action_predownload_highlighted(self) -> None:
        selected = self._highlighted_ranked_model()
        if selected is None:
            self.app.notify("Highlight a model in Model ranking first.", severity="warning", timeout=5)
            return
        revision = self.query_one("#model-revision", Input).value.strip() or None
        self.app.begin_storage_predownload(  # type: ignore[attr-defined]
            backend=BackendType.VLLM,
            model_id=selected.repo_id,
            revision=revision,
        )

    def _focus_model_list_if_pending(self) -> None:
        """Give the model picker focus once a requested ranking has loaded."""
        if not self._focus_model_list_when_loaded:
            return
        model_list = self.query_one(f"#{self._MODEL_LIST_ID}", OptionList)
        if model_list.option_count == 0:
            return
        self._focus_model_list_when_loaded = False
        model_list.focus()
        if model_list.highlighted is None:
            model_list.highlighted = 0

    def _set_model_status(self, text: str) -> None:
        self.query_one("#vllm-model-status", Static).update(text)

    def _set_ranking_title(self) -> None:
        subtitle = _ranking_subtitle(BackendType.VLLM, self._rank_mode)
        try:
            self.query_one("#vllm-model-ranking-title", Static).update(
                f"[bold]Model[/bold]  [dim]({subtitle})[/dim]"
            )
        except Exception:
            return

    def _resolve_rank_mode(self, option_id: str) -> str | None:
        if option_id == "rank-cached":
            return "cached"
        if option_id == "rank-downloads":
            return "downloads"
        if option_id == "rank-trending":
            return "trending"
        return None

    def _refresh_cached_models_from_storage(self, force: bool = False) -> None:
        refresher = getattr(self.app, "begin_storage_refresh", None)
        if callable(refresher):
            refresher(self, force=force)

    def _show_cached_models(self) -> None:
        self._ranked_models = list(self._cached_models)
        model_list = self.query_one("#vllm-model-list", OptionList)
        options = [Option(f"  {model.repo_id}", id=f"model-{idx}") for idx, model in enumerate(self._ranked_models)]
        _set_option_list(model_list, options)
        if self._ranked_models:
            self._set_model_status("[dim]Cached models loaded. Select one to prefill Model name.[/dim]")
            self._focus_model_list_if_pending()
        elif self._fall_back_to_downloads_ranking():
            return
        else:
            self._set_model_status("[yellow]No cached vLLM models found in storage.[/yellow]")

    def _fall_back_to_downloads_ranking(self) -> bool:
        """Switch an empty default "Cached in storage" view to "Most downloaded".

        A first-time user has nothing cached, so the default ranking mode opens
        the form on an empty picker and a warning. Only the automatic default
        is replaced: once the mode has been chosen by hand it is left alone.
        """
        if self._rank_mode_touched or self._rank_mode != "cached" or not self._has_cached_snapshot:
            return False
        self._rank_mode = "downloads"
        rank_mode_list = self.query_one("#vllm-rank-mode", OptionList)
        rank_mode_list.highlighted = 1
        self._set_ranking_title()
        self._set_model_status("[dim]Nothing cached yet. Loading popular models...[/dim]")
        self.app.begin_fetch_vllm_models("downloads", self)  # type: ignore[attr-defined]
        return True

    def _apply_ranked_model_selection(self, option_id: str) -> None:
        selected = _model_from_option_id(option_id, self._ranked_models)
        if selected is None:
            return
        self.query_one("#model-name", Input).value = selected.repo_id
        self._refresh_app_preview()
        self._refresh_vllm_memory_status()

    def _memory_cache_key(self, repo_id: str, revision: str | None) -> str:
        return f"{repo_id.strip().casefold()}@{(revision or '').strip().casefold()}"

    def _current_memory_lookup(self) -> tuple[str, str | None]:
        repo_id = self.query_one("#model-name", Input).value.strip()
        revision = self.query_one("#model-revision", Input).value.strip() or None
        return repo_id, revision

    def _set_vllm_memory_status(self, text: str) -> None:
        self.query_one("#vllm-vram-status", Static).update(text)

    def _current_tensor_parallel(self) -> int:
        raw = self.query_one("#n-gpu", Input).value.strip()
        try:
            parsed = int(raw)
        except ValueError:
            return 1
        return max(1, parsed)

    def _render_vllm_memory_status(self, estimate: VllmMemoryBreakdown | None) -> None:
        if estimate is None:
            self._set_vllm_memory_status("[dim]Estimated VRAM: N/A[/dim]")
            return
        tensor_parallel = self._current_tensor_parallel()
        per_gpu_gb = estimate.total_gb / max(1, tensor_parallel)
        # An image encoder's working memory is not modelled, so say so rather
        # than letting the total read as if it covered image requests.
        vision_note = (
            " — excludes unknown image working memory"
            if estimate.vision_working_memory_gb is None
            else ""
        )
        self._set_vllm_memory_status(
            "[dim]"
            f"Estimated VRAM (heuristic, ctx={estimate.context_tokens}): "
            f"~{estimate.total_gb:.1f} GB total, ~{per_gpu_gb:.1f} GB/GPU @ TP={tensor_parallel}"
            f"{vision_note}"
            "[/dim]"
        )

    def _refresh_vllm_memory_status(self, from_cache_only: bool = False) -> None:
        if not from_cache_only:
            self._cancel_vllm_memory_refresh()
        repo_id, revision = self._current_memory_lookup()
        if not repo_id:
            self._set_vllm_memory_status("[dim]Estimated VRAM: enter model name to compute[/dim]")
            return
        cache_key = self._memory_cache_key(repo_id, revision)
        if cache_key in self._model_to_memory_estimate:
            self._render_vllm_memory_status(self._model_to_memory_estimate[cache_key])
            return
        if from_cache_only:
            self._set_vllm_memory_status("[dim]Estimated VRAM: loading...[/dim]")
            return

        lookup_key = (repo_id.casefold(), (revision or "").casefold())
        if self._last_memory_lookup == lookup_key:
            return
        self._last_memory_lookup = lookup_key
        self._set_vllm_memory_status("[dim]Estimated VRAM: loading...[/dim]")
        self.run_worker(
            lambda: self._run_fetch_vllm_memory(repo_id=repo_id, revision=revision),
            name=f"vllm-memory-{repo_id}",
            thread=True,
        )

    def _cancel_vllm_memory_refresh(self) -> None:
        timer = self._memory_lookup_timer
        self._memory_lookup_timer = None
        if timer is not None:
            timer.stop()

    def _schedule_vllm_memory_refresh(self) -> None:
        """Debounce remote vLLM metadata lookups while the model field changes."""
        self._cancel_vllm_memory_refresh()
        repo_id, _revision = self._current_memory_lookup()
        if not repo_id:
            self._refresh_vllm_memory_status()
            return
        if not _is_plausible_model_lookup(repo_id):
            self._last_memory_lookup = None
            self._set_vllm_memory_status("[dim]Estimated VRAM: finish entering the model name[/dim]")
            return
        self._memory_lookup_timer = self.set_timer(
            _MODEL_LOOKUP_DEBOUNCE_SECONDS,
            self._run_scheduled_vllm_memory_refresh,
            name="vllm-memory-lookup-debounce",
        )

    def _run_scheduled_vllm_memory_refresh(self) -> None:
        self._memory_lookup_timer = None
        self._refresh_vllm_memory_status()

    def _run_fetch_vllm_memory(self, repo_id: str, revision: str | None) -> None:
        poster = getattr(self, "post_message", None)
        if poster is None:
            return
        try:
            estimate = fetch_vllm_memory_breakdown(repo_id=repo_id, revision=revision)
        except Exception as exc:
            poster(VllmMemoryFailed(repo_id=repo_id, revision=revision, error=str(exc)))
        else:
            poster(VllmMemoryLoaded(repo_id=repo_id, revision=revision, estimate=estimate))
        try:
            discover_reasoning_capabilities(
                BackendType.VLLM,
                repo_id,
                revision,
            )
        except Exception:
            pass

    def on_vllm_memory_loaded(self, message: VllmMemoryLoaded) -> None:
        cache_key = self._memory_cache_key(message.repo_id, message.revision)
        self._model_to_memory_estimate[cache_key] = message.estimate
        current_repo, current_revision = self._current_memory_lookup()
        if self._memory_cache_key(current_repo, current_revision) != cache_key:
            return
        self._render_vllm_memory_status(message.estimate)
        if self._prime_offers:
            self._refresh_prime_offer_options()

    def on_vllm_memory_failed(self, message: VllmMemoryFailed) -> None:
        current_repo, current_revision = self._current_memory_lookup()
        if not current_repo:
            return
        cache_key = self._memory_cache_key(message.repo_id, message.revision)
        if self._memory_cache_key(current_repo, current_revision) != cache_key:
            return
        self._model_to_memory_estimate.setdefault(cache_key, None)
        self._render_vllm_memory_status(None)
        if self._prime_offers:
            self._refresh_prime_offer_options()

    def _highlighted_ranked_model(self) -> ModelCandidate | None:
        highlighted = self.query_one("#vllm-model-list", OptionList).highlighted_option
        option_id = highlighted.id if highlighted is not None else ""
        return _model_from_option_id(option_id or "", self._ranked_models)

    def _refresh_app_preview(self) -> None:
        model_name = self.query_one("#model-name", Input).value.strip()
        instance_override = self.query_one("#instance-name-vllm", Input).value.strip()
        app_override = self.query_one("#app-name-vllm", Input).value.strip()
        if app_override:
            preview = app_override
        elif instance_override:
            preview = build_deployment_name(self._provider, BackendType.VLLM, instance_override)
        else:
            preview = build_deployment_name(
                self._provider,
                BackendType.VLLM,
                auto_instance_name_for_backend(BackendType.VLLM, model_name),
            )
        self.query_one("#vllm-app-preview", Static).update(f"[dim]App name preview: {escape(preview)}[/dim]")

    def _do_deploy(self) -> None:
        config = DeploymentConfig(backend=BackendType.VLLM, provider=self._provider)
        config.model_name = self.query_one("#model-name", Input).value.strip() or None
        if not config.model_name:
            self.app.notify(
                "Enter a model name before deploying.",
                severity="error",
                timeout=5,
            )
            self.query_one("#model-name", Input).focus()
            return
        config.model_revision = self.query_one("#model-revision", Input).value.strip() or None
        config.required_vram_gb = self._current_vllm_required_vram()
        gpu_type = normalize_gpu_type(self._selected_gpu_type)
        if not gpu_type:
            self.app.notify("GPU type is required.", severity="error", timeout=5)
            return
        gpu_count = parse_gpu_count(self.query_one("#gpu-count-vllm", Input).value, default=0)
        if gpu_count <= 0:
            self.app.notify("GPUs attached must be an integer >= 1.", severity="error", timeout=5)
            return
        config.gpu_type = gpu_type
        config.gpu_count = gpu_count
        if self._provider == ComputeProvider.PRIME:
            offer = self._prime_offers.get(self._selected_prime_offer_id or "")
            if offer is None:
                self.app.notify("Select a live Prime GPU offer.", severity="error", timeout=5)
                return
            config.gpu_type = offer.gpu_type
            config.gpu_count = offer.gpu_count
            allow_insecure_http = self.query_one("#prime-insecure-http", Switch).value
            config.provider_options = PrimeProviderOptions(
                offer_id=offer.id,
                disk_id=self.query_one("#prime-disk-id", Input).value.strip() or None,
                allow_insecure_http=allow_insecure_http,
                keep_failed_resource=self.query_one("#prime-keep-failed", Switch).value,
                auto_disk=self.query_one("#prime-auto-disk", Switch).value,
            )
        elif self._provider == ComputeProvider.VAST:
            vast_options = self._build_vast_provider_options("#vast-disk-vllm")
            if vast_options is None:
                return
            config.provider_options = vast_options
            vast_offer = self._vast_offers[vast_options.offer_id]
            config.gpu_type = vast_offer.gpu_type
            config.gpu_count = vast_offer.gpu_count
        alias = self.query_one("#served-model-name", Input).value.strip()
        config.served_model_name = alias or default_served_model_name(config.model_name)
        config.fast_boot = self.query_one("#fast-boot", Switch).value
        config.trust_remote_code = self.query_one("#trust-remote-code", Switch).value
        if self._provider in (ComputeProvider.PRIME, ComputeProvider.VAST):
            config.n_gpu = config.gpu_count
        else:
            n_gpu_str = self.query_one("#n-gpu", Input).value.strip()
            config.n_gpu = positive_int(n_gpu_str or "1")
            if config.n_gpu is None:
                self.app.notify("Tensor parallel size must be an integer >= 1.", severity="error", timeout=5)
                return
        config.tool_call_parser = self.query_one("#tool-call-parser", Input).value.strip() or None
        # Advanced values are always read: collapsing the section must never
        # silently discard options the user entered before collapsing it.
        config.reasoning_parser = self.query_one("#reasoning-parser", Input).value.strip() or None
        kwargs_raw = self.query_one("#chat-template-kwargs", Input).value.strip()
        if kwargs_raw:
            try:
                parsed = json.loads(kwargs_raw)
            except json.JSONDecodeError:
                self.app.notify(
                    "Default chat template kwargs must be valid JSON.",
                    severity="error",
                    timeout=6,
                )
                return
            if not isinstance(parsed, dict):
                self.app.notify(
                    "Default chat template kwargs must be a JSON object.",
                    severity="error",
                    timeout=6,
                )
                return
            config.default_chat_template_kwargs = kwargs_raw
        smoke_only = self.query_one("#smoke-only-vllm", Switch).value
        if smoke_only and self._provider != ComputeProvider.MODAL:
            self.app.notify(
                f"{self._provider.display_name} does not support smoke-test-only mode.",
                severity="error",
                timeout=5,
            )
            return
        config.do_deploy = not smoke_only
        config.run_smoke = smoke_only
        config.do_warmup = self.query_one("#warmup-vllm", Switch).value if config.do_deploy else False
        config.show_debug_logs = self.query_one("#show-debug-logs-vllm", Switch).value
        instance_override = self.query_one("#instance-name-vllm", Input).value.strip()
        app_override = self.query_one("#app-name-vllm", Input).value.strip()
        if app_override:
            config.app_name = app_override
            config.instance_name = slugify_instance_name(instance_override or app_override)
        elif instance_override:
            config.instance_name = slugify_instance_name(instance_override)
            config.app_name = build_deployment_name(
                config.provider, config.backend, config.instance_name
            )
        else:
            config.instance_name = auto_instance_name_for_backend(config.backend, config.model_name)
            config.app_name = build_deployment_name(
                config.provider, config.backend, config.instance_name
            )

        try:
            self.query_one(VisionOptions).apply(config)
        except ValueError as exc:
            self.app.notify(str(exc), severity="error", timeout=8)
            return
        # Refuse here rather than after routing, so the form never accepts a
        # configuration its provider will reject.
        reason = refuse_deployment(config)
        if reason:
            self.app.notify(reason, severity="error", timeout=8)
            return
        self.app.begin_deploy(config)  # type: ignore[attr-defined]

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_open_storage(self) -> None:
        self.app.action_push_storage(BackendType.VLLM)  # type: ignore[attr-defined]
