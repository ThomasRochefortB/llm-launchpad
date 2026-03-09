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
from ...core.modal_gpu import fetch_modal_gpu_types
from ...core.naming import (
    auto_instance_name_for_backend,
    build_app_name,
    default_served_model_name,
    slugify_instance_name,
)
from ...protocol.enums import BackendType
from ...protocol.models import DeploymentConfig, StorageSnapshot
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


class GpuTypesLoaded(Message):
    """GPU type options were fetched successfully."""

    def __init__(self, gpu_types: list[str]) -> None:
        super().__init__()
        self.gpu_types = gpu_types


class GpuTypesFailed(Message):
    """GPU type option fetch failed."""

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


def _quant_preview(quantizations: list[str], vram_gb_by_quant: dict[str, float], limit: int = 6) -> str:
    preview_tokens = [_format_quant_with_vram(quant, vram_gb_by_quant) for quant in quantizations[:limit]]
    suffix = "..." if len(quantizations) > limit else ""
    return ", ".join(preview_tokens) + suffix


def _advance_deploy_focus(screen: CopyEnabledScreen, navigation_order: tuple[str, ...]) -> None:
    """Advance focus to the next visible deploy form widget."""
    move_focus_across_widgets(
        screen,
        navigation_order,
        direction=1,
        is_focusable=_is_focusable_for_arrow_navigation,
    )


class BackendSelectScreen(CopyEnabledScreen):
    """Step 1: pick backend (llama.cpp or vLLM)."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-container"):
            yield Static("[bold cyan]Deploy[/bold cyan]  [dim]Step 1: Choose backend[/dim]")
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


class LlamaCppDeployScreen(CopyEnabledScreen):
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
        "gpu-type-llama",
        "gpu-count-llama",
        "preload",
        "do-deploy",
        "warmup",
        "toggle-advanced-llama",
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
        with VerticalScroll(id="menu-container"):
            yield Static("[bold cyan]Deploy llama.cpp[/bold cyan]  [dim]Step 2: Model & options[/dim]")
            yield Static("")

            yield Static("[bold]Model ranking[/bold]  [dim](Top 10 GGUF text-generation models)[/dim]")
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

            # Options
            with Vertical(classes="gpu-config-panel"):
                yield Static("GPU configuration", classes="form-section-title")
                yield Static("Select the deployment GPU shape.", classes="form-section-subtitle")
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
            yield ToggleField("Preload/download weights now", "preload", default=True)
            yield ToggleField("Deploy the server", "do-deploy", default=True)
            yield ToggleField("Warm up after deploy", "warmup", default=True)

            yield Static("")

            # Advanced options (collapsed by default)
            yield Button("Advanced options...", id="toggle-advanced-llama", variant="default")
            yield FormField("HF revision (optional)", "revision", classes="llama-advanced")
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
            yield FormField(
                "Instance name (optional)",
                "instance-name-llama",
                hint="Auto-derived from repo if blank",
                classes="llama-advanced",
            )
            yield FormField(
                "App name override (optional)",
                "app-name-llama",
                hint="Advanced: explicit Modal app name",
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
        self._last_quant_lookup: tuple[str, str] | None = None
        self._updating_quant_input = False
        self._quant_touched = False
        self._selected_gpu_type = DEFAULT_GPU_TYPE
        for widget in self.query(".llama-advanced"):
            widget.add_class("hidden")
        rank_mode_list = self.query_one("#llama-rank-mode", OptionList)
        if rank_mode_list.option_count > 0:
            rank_mode_list.highlighted = 0
        rank_mode_list.focus()
        self._refresh_gpu_types()
        self._set_model_status("[dim]Loading cached models from storage...[/dim]")
        self._refresh_cached_models_from_storage()
        self._refresh_app_preview()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "llama-rank-mode":
            selected_mode = self._resolve_rank_mode(event.option.id or "")
            if selected_mode is None:
                return
            if selected_mode != self._rank_mode:
                self._rank_mode = selected_mode
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
        elif event.button.id == "deploy-btn":
            self._do_deploy()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "quant" and not self._updating_quant_input:
            self._quant_touched = True
        if event.input.id in {"repo-id", "instance-name-llama", "app-name-llama"}:
            self._refresh_app_preview()
        if event.input.id in {"repo-id", "revision"}:
            self._lookup_quantizations_for_current_repo()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "gpu-type-llama":
            return
        if isinstance(event.value, str) and event.value.strip():
            self._selected_gpu_type = normalize_gpu_type(event.value)

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
            gpu_types = fetch_modal_gpu_types()
        except Exception as exc:
            poster(GpuTypesFailed(error=str(exc)))
            return
        poster(GpuTypesLoaded(gpu_types=gpu_types))

    def on_gpu_types_loaded(self, message: GpuTypesLoaded) -> None:
        dropdown = self.query_one("#gpu-type-llama", Select)
        options = build_gpu_type_options(message.gpu_types)
        if self._selected_gpu_type and self._selected_gpu_type not in options:
            options.insert(0, self._selected_gpu_type)
        if not options:
            return
        dropdown.set_options([(value, value) for value in options])
        selected = self._selected_gpu_type if self._selected_gpu_type in options else options[0]
        dropdown.value = selected
        self._selected_gpu_type = selected

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
        self._apply_quantizations(
            self._display_quantizations_for_repo(repo_key, list(message.quantizations)),
            auto_select=not self._quant_touched,
            vram_gb_by_quant=self._repo_to_quant_vram[repo_key],
        )

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
        config = DeploymentConfig(backend=BackendType.LLAMACPP)
        config.repo_id = self.query_one("#repo-id", Input).value.strip() or None
        config.quant = self.query_one("#quant", Input).value.strip() or None
        rev = self.query_one("#revision", Input).value.strip()
        config.revision = rev or None

        config.preload = self.query_one("#preload", Switch).value
        config.do_deploy = self.query_one("#do-deploy", Switch).value
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

        # Advanced
        adv_visible = not self.query(".llama-advanced").first().has_class("hidden")
        if adv_visible:
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
            config.app_name = build_app_name(config.backend, config.instance_name)
        else:
            config.instance_name = auto_instance_name_for_backend(config.backend, model_hint)
            config.app_name = build_app_name(config.backend, config.instance_name)

        self.app.begin_deploy(config)  # type: ignore[attr-defined]

    def _set_model_status(self, text: str) -> None:
        self.query_one("#llama-model-status", Static).update(text)

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
            preview = build_app_name(BackendType.LLAMACPP, instance_override)
        else:
            preview = build_app_name(
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


class VllmDeployScreen(CopyEnabledScreen):
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
        "toggle-advanced-vllm",
        "model-revision",
        "smoke-only-vllm",
        "warmup-vllm",
        "fast-boot",
        "trust-remote-code",
        "show-debug-logs-vllm",
        "served-model-name",
        "reasoning-parser",
        "chat-template-kwargs",
        "instance-name-vllm",
        "app-name-vllm",
        "deploy-vllm-btn",
    )

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="menu-container"):
            yield Static("[bold cyan]Deploy vLLM[/bold cyan]  [dim]Step 2: Model & options[/dim]")
            yield Static("")

            yield Static("[bold]Model ranking[/bold]  [dim](Top 10 text-generation models)[/dim]")
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
            with Vertical(classes="gpu-config-panel"):
                yield Static("GPU configuration", classes="form-section-title")
                yield Static(
                    "Choose deployment GPUs and in-replica tensor sharding.",
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
                        yield Static("Deployment GPU count", classes="form-label")
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
                    hint="vLLM --tensor-parallel-size (separate from deployment GPU count)",
                    classes="gpu-config-tensor-field",
                )
            yield Button("Advanced options...", id="toggle-advanced-vllm", variant="default")
            yield FormField(
                "Model revision (optional)",
                "model-revision",
                hint="Leave blank to use default branch",
                classes="vllm-advanced",
            )
            yield ToggleField(
                "Smoke test only (no deploy)",
                "smoke-only-vllm",
                default=False,
                classes="vllm-advanced",
            )
            yield ToggleField(
                "Verify readiness after deploy",
                "warmup-vllm",
                default=True,
                classes="vllm-advanced",
            )
            yield ToggleField("Enforce eager startup", "fast-boot", default=False, classes="vllm-advanced")
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
            yield FormField(
                "Instance name (optional)",
                "instance-name-vllm",
                hint="Auto-derived from model if blank",
                classes="vllm-advanced",
            )
            yield FormField(
                "App name override (optional)",
                "app-name-vllm",
                hint="Advanced: explicit Modal app name",
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
        self._served_alias_touched = False
        self._updating_served_alias = False
        self._last_auto_served_alias = ""
        self._model_to_memory_estimate: dict[str, VllmMemoryBreakdown | None] = {}
        self._last_memory_lookup: tuple[str, str] | None = None
        for widget in self.query(".vllm-advanced"):
            widget.add_class("hidden")
        rank_mode_list = self.query_one("#vllm-rank-mode", OptionList)
        if rank_mode_list.option_count > 0:
            rank_mode_list.highlighted = 0
        rank_mode_list.focus()
        self._refresh_gpu_types()
        self._set_model_status("[dim]Loading cached models from storage...[/dim]")
        self._refresh_cached_models_from_storage()
        self._sync_served_alias_from_model(force=True)
        self._refresh_app_preview()
        self._refresh_vllm_memory_status()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "vllm-rank-mode":
            selected_mode = self._resolve_rank_mode(event.option.id or "")
            if selected_mode is None:
                return
            if selected_mode != self._rank_mode:
                self._rank_mode = selected_mode
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
        elif event.button.id == "deploy-vllm-btn":
            self._do_deploy()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-name":
            self._sync_served_alias_from_model()
            self._refresh_vllm_memory_status()
        elif event.input.id == "model-revision":
            self._refresh_vllm_memory_status()
        elif event.input.id == "n-gpu":
            self._refresh_vllm_memory_status(from_cache_only=True)
        elif event.input.id == "served-model-name" and not self._updating_served_alias:
            self._served_alias_touched = event.input.value.strip() != self._last_auto_served_alias
        if event.input.id in {"model-name", "instance-name-vllm", "app-name-vllm"}:
            self._refresh_app_preview()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "gpu-type-vllm":
            return
        if isinstance(event.value, str) and event.value.strip():
            self._selected_gpu_type = normalize_gpu_type(event.value)

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
            gpu_types = fetch_modal_gpu_types()
        except Exception as exc:
            poster(GpuTypesFailed(error=str(exc)))
            return
        poster(GpuTypesLoaded(gpu_types=gpu_types))

    def on_gpu_types_loaded(self, message: GpuTypesLoaded) -> None:
        dropdown = self.query_one("#gpu-type-vllm", Select)
        options = build_gpu_type_options(message.gpu_types)
        if self._selected_gpu_type and self._selected_gpu_type not in options:
            options.insert(0, self._selected_gpu_type)
        if not options:
            return
        dropdown.set_options([(value, value) for value in options])
        selected = self._selected_gpu_type if self._selected_gpu_type in options else options[0]
        dropdown.value = selected
        self._selected_gpu_type = selected

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

    def _run_fetch_vllm_memory(self, repo_id: str, revision: str | None) -> None:
        poster = getattr(self, "post_message", None)
        if poster is None:
            return
        try:
            estimate = fetch_vllm_memory_breakdown(repo_id=repo_id, revision=revision)
        except Exception as exc:
            poster(VllmMemoryFailed(repo_id=repo_id, revision=revision, error=str(exc)))
            return
        poster(VllmMemoryLoaded(repo_id=repo_id, revision=revision, estimate=estimate))

    def on_vllm_memory_loaded(self, message: VllmMemoryLoaded) -> None:
        cache_key = self._memory_cache_key(message.repo_id, message.revision)
        self._model_to_memory_estimate[cache_key] = message.estimate
        current_repo, current_revision = self._current_memory_lookup()
        if self._memory_cache_key(current_repo, current_revision) != cache_key:
            return
        self._render_vllm_memory_status(message.estimate)

    def on_vllm_memory_failed(self, _: VllmMemoryFailed) -> None:
        current_repo, current_revision = self._current_memory_lookup()
        if not current_repo:
            return
        cache_key = self._memory_cache_key(current_repo, current_revision)
        self._model_to_memory_estimate.setdefault(cache_key, None)
        self._render_vllm_memory_status(None)

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
            preview = build_app_name(BackendType.VLLM, instance_override)
        else:
            preview = build_app_name(
                BackendType.VLLM,
                auto_instance_name_for_backend(BackendType.VLLM, model_name),
            )
        self.query_one("#vllm-app-preview", Static).update(f"[dim]App name preview: {preview}[/dim]")

    def _do_deploy(self) -> None:
        config = DeploymentConfig(backend=BackendType.VLLM)
        config.model_name = self.query_one("#model-name", Input).value.strip() or None
        config.model_revision = self.query_one("#model-revision", Input).value.strip() or None
        gpu_type = normalize_gpu_type(self._selected_gpu_type)
        if not gpu_type:
            self.app.notify("GPU type is required.", severity="error", timeout=5)
            return
        gpu_count = parse_gpu_count(self.query_one("#gpu-count-vllm", Input).value, default=0)
        if gpu_count <= 0:
            self.app.notify("Deployment GPU count must be an integer >= 1.", severity="error", timeout=5)
            return
        config.gpu_type = gpu_type
        config.gpu_count = gpu_count
        alias = self.query_one("#served-model-name", Input).value.strip()
        config.served_model_name = alias or default_served_model_name(config.model_name)
        config.fast_boot = self.query_one("#fast-boot", Switch).value
        config.trust_remote_code = self.query_one("#trust-remote-code", Switch).value
        n_gpu_str = self.query_one("#n-gpu", Input).value.strip()
        try:
            config.n_gpu = int(n_gpu_str) if n_gpu_str else 1
        except ValueError:
            config.n_gpu = 1
        adv_visible = not self.query(".vllm-advanced").first().has_class("hidden")
        if adv_visible:
            config.reasoning_parser = self.query_one("#reasoning-parser", Input).value.strip() or None
            config.tool_call_parser = self.query_one("#tool-call-parser", Input).value.strip() or None
        kwargs_raw = self.query_one("#chat-template-kwargs", Input).value.strip() if adv_visible else ""
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
            config.app_name = build_app_name(config.backend, config.instance_name)
        else:
            config.instance_name = auto_instance_name_for_backend(config.backend, config.model_name)
            config.app_name = build_app_name(config.backend, config.instance_name)

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
