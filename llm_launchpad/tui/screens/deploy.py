"""Deploy screen: backend selection, model config, deploy options.

Two sub-flows: llama.cpp (ranked GGUF/custom) and vLLM (model params).
Keyboard-driven form navigation with enter-to-proceed.
"""

from __future__ import annotations

import json

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
from ...protocol.enums import BackendType, ComputeProvider
from ...protocol.models import (
    ComputeOffer,
    DeploymentConfig,
    PrimeProviderOptions,
    StorageSnapshot,
)
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
        location = offer.country or offer.region or offer.data_center or "-"
        price = (
            f"${offer.price_per_hour:.3f}/hr"
            if offer.price_per_hour is not None
            else "price n/a"
        )
        options.append(
            (
                f"{offer.id} · {offer.gpu_count}x {offer.gpu_type} · "
                f"{location} · {price}",
                offer.id,
            )
        )
    return options


def _prime_offer_status(
    offer_count: int,
    required_vram_gb: float | None,
    backend: BackendType = BackendType.LLAMACPP,
) -> str:
    strategy = f"portable {preferred_prime_offer_image(backend)}"
    if required_vram_gb is None:
        if offer_count:
            return (
                f"[dim]{offer_count} live {strategy} GPU offers. "
                "Model VRAM is not known yet; "
                "choose a model to narrow them.[/dim]"
            )
        return (
            f"[yellow]No secure on-demand {strategy} GPU offers are "
            "currently available.[/yellow]"
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


def _advance_deploy_focus(screen: CopyEnabledScreen, navigation_order: tuple[str, ...]) -> None:
    """Advance focus to the next visible deploy form widget."""
    move_focus_across_widgets(
        screen,
        navigation_order,
        direction=1,
        is_focusable=_is_focusable_for_arrow_navigation,
    )


class _CostPreviewMixin:
    """Shared hourly cost preview behavior for custom deploy forms."""

    _gpu_price_by_value: dict[str, float]
    _provider: ComputeProvider
    _prime_offers: dict[str, ComputeOffer]
    _selected_prime_offer_id: str | None
    _selected_gpu_type: str | None

    def _current_gpu_hourly_price(self) -> float | None:
        if self._provider == ComputeProvider.PRIME:
            offer = self._prime_offers.get(self._selected_prime_offer_id or "")
            return offer.price_per_hour if offer is not None else None
        return self._gpu_price_by_value.get((self._selected_gpu_type or "").strip())

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
                Option("  llama.cpp (GGUF) ( Recommended )", id="llamacpp"),
                Option("  vLLM", id="vllm"),
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


class LlamaCppDeployScreen(_CostPreviewMixin, CopyEnabledScreen):
    """llama.cpp deploy form."""

    BINDINGS = [
        Binding("up", "navigate_option_list_up", show=False, priority=True),
        Binding("down", "navigate_option_list_down", show=False, priority=True),
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+d", "do_deploy", "Deploy", show=True),
        Binding("ctrl+s", "open_storage", "Storage", show=True),
        Binding("p", "predownload_highlighted", "Pre-download", show=True),
    ]
    NAVIGATION_ORDER = (
        "llama-rank-mode",
        "llama-model-list",
        "repo-id",
        "quant",
        "llama-quant-list",
        "provider-llama",
        "prime-offer-llama",
        "gpu-type-llama",
        "gpu-count-llama",
        "toggle-advanced-llama",
        "warmup",
        "revision",
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
                options=[("Modal", "modal"), ("Prime Intellect", "prime")],
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

            # Options
            with Vertical(classes="gpu-config-panel"):
                yield Static("GPU configuration", classes="form-section-title")
                yield Static(
                    "Select a Modal GPU shape or bind an exact live Prime offer.",
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
            yield Static("[bold]Runtime[/bold]", classes="llama-advanced")
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
            yield Static("[bold]Naming[/bold]", classes="llama-advanced")
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
        self._provider = ComputeProvider.MODAL
        self._prime_offers: dict[str, ComputeOffer] = {}
        self._selected_prime_offer_id: str | None = None
        self._gpu_price_by_value: dict[str, float] = {}
        for widget in self.query(".llama-advanced"):
            widget.add_class("hidden")
        for widget in self.query(".prime-only"):
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
            if selected_mode != self._rank_mode:
                self._rank_mode = selected_mode
                self._set_ranking_title()
                self._ranked_models = []
                self._set_model_status("[dim]Loading model suggestions...[/dim]")
                self.query_one("#llama-model-list", OptionList).set_options([])
                if self._rank_mode == "cached":
                    self._set_model_status("[dim]Loading cached models from storage...[/dim]")
                    if self._has_cached_snapshot:
                        self._show_cached_models()
                    self._refresh_cached_models_from_storage()
                else:
                    self.app.begin_fetch_llamacpp_models(self._rank_mode, self)  # type: ignore[attr-defined]
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
            elif self._provider == ComputeProvider.MODAL:
                self._refresh_gpu_types()
            self._refresh_app_preview()
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
        self._update_cost_preview("llama-cost-preview")

    def _sync_prime_visibility(self) -> None:
        advanced_visible = not self.query_one("#warmup").has_class("hidden")
        for widget in self.query(".prime-only"):
            hide = self._provider != ComputeProvider.PRIME
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
            f"[red]Could not load Prime offers:[/red] {message.error}"
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
            label = f"  {model.repo_id:<38} downloads={downloads:<10} likes={likes}"
            options.append(Option(label, id=f"model-{idx}"))
        model_list.set_options(options)

        if message.models:
            mode_label = "Most downloaded" if self._rank_mode == "downloads" else "Trending"
            self._set_model_status(f"[dim]{mode_label} models loaded. Select one to prefill repo-id.[/dim]")
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
        self.query_one("#llama-model-list", OptionList).set_options([])
        self._set_model_status(f"[yellow]Could not load cached models:[/yellow] {message.error}")

    def on_llama_cpp_models_failed(self, message: LlamaCppModelsFailed) -> None:
        if message.mode != self._rank_mode:
            return
        self._ranked_models = []
        self._repo_to_quants = {}
        self.query_one("#llama-model-list", OptionList).set_options([])
        self._set_model_status(
            f"[yellow]Could not load model suggestions:[/yellow] {message.error} [dim](manual input still works)[/dim]"
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
        architecture_label = message.architecture or "unknown"
        if message.compatibility_status == "supported":
            self._set_quant_status(
                "[dim]Quantizations:[/dim] "
                f"[green]Compatible architecture={architecture_label}[/green]"
            )
        elif message.compatibility_status == "unsupported":
            self._set_quant_status(
                "[dim]Quantizations:[/dim] "
                f"[red]Unsupported architecture {architecture_label}:[/red] "
                f"{message.compatibility_message}"
            )
        else:
            detail = (
                message.compatibility_message
                or "GGUF architecture metadata was not available."
            )
            self._set_quant_status(
                f"[dim]Quantizations:[/dim] "
                f"[yellow]Compatibility unknown:[/yellow] {detail}"
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
        self.query_one("#llama-quant-list", OptionList).set_options([])
        self._set_quant_status(
            f"[yellow]Could not load quantizations:[/yellow] {message.error} [dim](manual quant still works)[/dim]"
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

        # Advanced values are always read: collapsing the section must never
        # silently discard options the user entered before collapsing it.
        sa = self.query_one("#server-args", Input).value.strip()
        config.server_args = sa or None
        h = self.query_one("#host-input", Input).value.strip()
        config.host = h or None
        p = self.query_one("#port-input", Input).value.strip()
        if p:
            try:
                config.port = int(p)
            except ValueError:
                pass
        ngl = self.query_one("#n-gpu-layers", Input).value.strip()
        if ngl:
            try:
                config.n_gpu_layers = int(ngl)
            except ValueError:
                pass
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

        self.app.begin_deploy(config)  # type: ignore[attr-defined]

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
            label = f"  {model.repo_id:<38} quants={quant_preview}"
            options.append(Option(label, id=f"model-{idx}"))
        model_list.set_options(options)
        if self._ranked_models:
            self._set_model_status("[dim]Cached models loaded. Select one to prefill repo-id.[/dim]")
        else:
            self._set_model_status("[yellow]No cached llama.cpp models found in storage.[/yellow]")

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
                list(selected.quantizations),
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
        if self._rank_mode != "cached":
            return list(quantizations)
        cached_quants = self._cached_repo_to_quants.get(repo_key)
        if not cached_quants:
            return list(quantizations)
        allowed = {quant.strip().upper() for quant in cached_quants}
        filtered = [quant for quant in quantizations if quant.strip().upper() in allowed]
        if filtered:
            return filtered
        return list(cached_quants)

    def _lookup_quantizations_for_current_repo(self, force_refresh: bool = False) -> None:
        self._cancel_quantization_lookup()
        repo_id = self.query_one("#repo-id", Input).value.strip()
        revision = self.query_one("#revision", Input).value.strip()
        if not repo_id:
            self.query_one("#llama-quant-list", OptionList).set_options([])
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
        options = [
            Option(f"  {_format_quant_with_vram(quant, normalized_vram)}", id=f"quant-{quant}")
            for quant in sorted_quantizations
        ]
        quant_list.set_options(options)
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
        self.query_one("#llama-app-preview", Static).update(f"[dim]App name preview: {preview}[/dim]")

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_navigate_option_list_down(self) -> None:
        if move_focus_across_option_lists(
            self,
            ("llama-rank-mode", "llama-model-list", "llama-quant-list"),
            direction=1,
        ):
            return
        if move_focus_across_widgets(
            self,
            self.NAVIGATION_ORDER,
            direction=1,
            is_focusable=_is_focusable_for_arrow_navigation,
        ):
            return
        raise SkipAction()

    def action_navigate_option_list_up(self) -> None:
        if move_focus_across_option_lists(
            self,
            ("llama-rank-mode", "llama-model-list", "llama-quant-list"),
            direction=-1,
        ):
            return
        if move_focus_across_widgets(
            self,
            self.NAVIGATION_ORDER,
            direction=-1,
            is_focusable=_is_focusable_for_arrow_navigation,
        ):
            return
        raise SkipAction()

    def action_open_storage(self) -> None:
        self.app.action_push_storage(BackendType.LLAMACPP)  # type: ignore[attr-defined]


class VllmDeployScreen(_CostPreviewMixin, CopyEnabledScreen):
    """vLLM deploy form."""

    BINDINGS = [
        Binding("up", "navigate_option_list_up", show=False, priority=True),
        Binding("down", "navigate_option_list_down", show=False, priority=True),
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+d", "do_deploy", "Deploy", show=True),
        Binding("ctrl+s", "open_storage", "Storage", show=True),
        Binding("p", "predownload_highlighted", "Pre-download", show=True),
    ]
    NAVIGATION_ORDER = (
        "vllm-rank-mode",
        "vllm-model-list",
        "model-name",
        "gpu-type-vllm",
        "gpu-count-vllm",
        "n-gpu",
        "provider-vllm",
        "prime-offer-vllm",
        "toggle-advanced-vllm",
        "model-revision",
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
                options=[("Modal", "modal"), ("Prime Intellect", "prime")],
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
            with Vertical(classes="gpu-config-panel"):
                yield Static("GPU configuration", classes="form-section-title")
                yield Static(
                    "Choose deployment GPUs and in-replica tensor sharding. Base Modal hourly price per GPU is shown when available.",
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
                        yield Static("GPU count", classes="form-label")
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
                    hint="vLLM --tensor-parallel-size (separate from GPU count)",
                    classes="gpu-config-tensor-field",
                )
                yield Static("", id="vllm-cost-preview")
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
            yield Static("[bold]Runtime[/bold]", classes="vllm-advanced")
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
            yield Static("[bold]Naming[/bold]", classes="vllm-advanced")
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
            yield Static("[dim]App name preview: auto[/dim]", id="vllm-app-preview")

            yield Static("")
            yield Button("Deploy", id="deploy-vllm-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self._rank_mode = "cached"
        self._ranked_models: list[ModelCandidate] = []
        self._cached_models: list[ModelCandidate] = []
        self._has_cached_snapshot = False
        self._selected_gpu_type = DEFAULT_GPU_TYPE
        self._provider = ComputeProvider.MODAL
        self._prime_offers: dict[str, ComputeOffer] = {}
        self._selected_prime_offer_id: str | None = None
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
        for widget in self.query(".vllm-advanced"):
            widget.add_class("hidden")
        for widget in self.query(".prime-only"):
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
            if selected_mode != self._rank_mode:
                self._rank_mode = selected_mode
                self._set_ranking_title()
                self._ranked_models = []
                self._set_model_status("[dim]Loading model suggestions...[/dim]")
                self.query_one("#vllm-model-list", OptionList).set_options([])
                if self._rank_mode == "cached":
                    self._set_model_status("[dim]Loading cached models from storage...[/dim]")
                    if self._has_cached_snapshot:
                        self._show_cached_models()
                    self._refresh_cached_models_from_storage()
                else:
                    self.app.begin_fetch_vllm_models(self._rank_mode, self)  # type: ignore[attr-defined]
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
            elif self._provider == ComputeProvider.MODAL:
                self._refresh_gpu_types()
            self._refresh_app_preview()
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
        self._update_cost_preview("vllm-cost-preview")

    def _sync_prime_visibility(self) -> None:
        advanced_visible = not self.query_one("#model-revision").has_class("hidden")
        for widget in self.query(".prime-only"):
            hide = self._provider != ComputeProvider.PRIME
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
            f"[red]Could not load Prime offers:[/red] {message.error}"
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
            label = f"  {model.repo_id:<38} downloads={downloads:<10} likes={likes}"
            options.append(Option(label, id=f"model-{idx}"))
        model_list.set_options(options)

        if message.models:
            mode_label = "Most downloaded" if self._rank_mode == "downloads" else "Trending"
            self._set_model_status(f"[dim]{mode_label} models loaded. Select one to prefill Model name.[/dim]")
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
        self.query_one("#vllm-model-list", OptionList).set_options([])
        self._set_model_status(f"[yellow]Could not load cached models:[/yellow] {message.error}")

    def on_vllm_models_failed(self, message: VllmModelsFailed) -> None:
        if message.mode != self._rank_mode:
            return
        self._ranked_models = []
        self.query_one("#vllm-model-list", OptionList).set_options([])
        self._set_model_status(
            f"[yellow]Could not load model suggestions:[/yellow] {message.error} [dim](manual input still works)[/dim]"
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
        model_list.set_options(options)
        if self._ranked_models:
            self._set_model_status("[dim]Cached models loaded. Select one to prefill Model name.[/dim]")
        else:
            self._set_model_status("[yellow]No cached vLLM models found in storage.[/yellow]")

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
        self._set_vllm_memory_status(
            "[dim]"
            f"Estimated VRAM (heuristic, ctx={estimate.context_tokens}): "
            f"~{estimate.total_gb:.1f} GB total, ~{per_gpu_gb:.1f} GB/GPU @ TP={tensor_parallel}"
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
        self.query_one("#vllm-app-preview", Static).update(f"[dim]App name preview: {preview}[/dim]")

    def _do_deploy(self) -> None:
        config = DeploymentConfig(backend=BackendType.VLLM, provider=self._provider)
        config.model_name = self.query_one("#model-name", Input).value.strip() or None
        config.model_revision = self.query_one("#model-revision", Input).value.strip() or None
        config.required_vram_gb = self._current_vllm_required_vram()
        gpu_type = normalize_gpu_type(self._selected_gpu_type)
        if not gpu_type:
            self.app.notify("GPU type is required.", severity="error", timeout=5)
            return
        gpu_count = parse_gpu_count(self.query_one("#gpu-count-vllm", Input).value, default=0)
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
        alias = self.query_one("#served-model-name", Input).value.strip()
        config.served_model_name = alias or default_served_model_name(config.model_name)
        config.fast_boot = self.query_one("#fast-boot", Switch).value
        config.trust_remote_code = self.query_one("#trust-remote-code", Switch).value
        if self._provider == ComputeProvider.PRIME:
            config.n_gpu = config.gpu_count
        else:
            n_gpu_str = self.query_one("#n-gpu", Input).value.strip()
            try:
                config.n_gpu = int(n_gpu_str) if n_gpu_str else 1
            except ValueError:
                config.n_gpu = 1
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
        if smoke_only and self._provider == ComputeProvider.PRIME:
            self.app.notify("Prime does not support smoke-test-only mode.", severity="error", timeout=5)
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

        self.app.begin_deploy(config)  # type: ignore[attr-defined]

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_navigate_option_list_down(self) -> None:
        if move_focus_across_option_lists(
            self,
            ("vllm-rank-mode", "vllm-model-list"),
            direction=1,
        ):
            return
        if move_focus_across_widgets(
            self,
            self.NAVIGATION_ORDER,
            direction=1,
            is_focusable=_is_focusable_for_arrow_navigation,
        ):
            return
        raise SkipAction()

    def action_navigate_option_list_up(self) -> None:
        if move_focus_across_option_lists(
            self,
            ("vllm-rank-mode", "vllm-model-list"),
            direction=-1,
        ):
            return
        if move_focus_across_widgets(
            self,
            self.NAVIGATION_ORDER,
            direction=-1,
            is_focusable=_is_focusable_for_arrow_navigation,
        ):
            return
        raise SkipAction()

    def action_open_storage(self) -> None:
        self.app.action_push_storage(BackendType.VLLM)  # type: ignore[attr-defined]
