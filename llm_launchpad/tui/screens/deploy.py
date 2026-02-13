"""Deploy screen: backend selection, model config, deploy options.

Two sub-flows: llama.cpp (preset/custom GGUF) and vLLM (model params).
Keyboard-driven form navigation with enter-to-proceed.
"""

from __future__ import annotations

import json

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
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

from ...core.hf_models import ModelCandidate
from ...core.modal_gpu import fetch_modal_gpu_types
from ...core.naming import (
    auto_instance_name_for_backend,
    build_app_name,
    default_served_model_name,
    slugify_instance_name,
)
from ...protocol.enums import BackendType
from ...protocol.models import DeploymentConfig
from ...presets import PRESETS
from ..gpu_config import (
    DEFAULT_GPU_COUNT,
    DEFAULT_GPU_TYPE,
    build_gpu_type_options,
    normalize_gpu_type,
    parse_gpu_count,
)
from ..workers import VllmModelsFailed, VllmModelsLoaded
from ..widgets.input_form import FormField, ToggleField


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


class BackendSelectScreen(Screen):
    """Step 1: pick backend (llama.cpp or vLLM)."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-container"):
            yield Static("[bold cyan]Deploy[/bold cyan]  [dim]Step 1: Choose backend[/dim]")
            yield Static("")
            yield OptionList(
                Option("  llama.cpp (GGUF)            Quantized models, single GPU", id="llamacpp"),
                Option("  vLLM (OpenAI-compatible)    Full-precision, tensor parallel", id="vllm"),
                id="backend-list",
            )
        yield Footer()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id == "llamacpp":
            self.app.push_screen(LlamaCppDeployScreen())
        elif event.option.id == "vllm":
            self.app.push_screen(VllmDeployScreen())

    def action_pop_screen(self) -> None:
        self.app.pop_screen()


class LlamaCppDeployScreen(Screen):
    """llama.cpp deploy form: preset selection + options."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+d", "do_deploy", "Deploy", show=True),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="menu-container"):
            yield Static("[bold cyan]Deploy llama.cpp[/bold cyan]  [dim]Step 2: Model & options[/dim]")
            yield Static("")

            # Preset selection
            yield Static("[bold]Select a preset[/bold]  [dim](or choose custom)[/dim]")
            options = []
            for name, entry in PRESETS.items():
                label = f"  {name:<24} {entry.get('repo_id', '')}  [{entry.get('quant', '')}]"
                options.append(Option(label, id=f"preset-{name}"))
            options.append(Option("  custom                     Enter repo-id and quant manually", id="preset-custom"))
            yield OptionList(*options, id="preset-list")

            # Custom fields (hidden by default, shown when custom selected)
            with Vertical(id="custom-fields", classes="hidden"):
                yield FormField(
                    "Hugging Face repo-id",
                    "repo-id",
                    hint="e.g., Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                )
                yield FormField("Quant pattern", "quant", default="Q4_K_M")
                yield FormField("HF revision (optional)", "revision")

            yield Static("")

            # Options
            yield Static("GPU configuration", classes="form-label")
            with Horizontal(id="gpu-config-row-llama"):
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
            yield Button("Advanced options...", id="toggle-advanced", variant="default")
            with Vertical(id="advanced-fields", classes="hidden"):
                yield FormField("Server args", "server-args", hint="e.g., --ctx-size 65536")
                yield FormField("Host", "host-input", default="0.0.0.0")
                yield FormField("Port", "port-input", default="8080")
                yield FormField("n_gpu_layers (blank=auto)", "n-gpu-layers")

            yield Static("")
            yield FormField(
                "Instance name (optional)",
                "instance-name-llama",
                hint="Auto-derived from preset/repo if blank",
            )
            yield FormField(
                "App name override (optional)",
                "app-name-llama",
                hint="Advanced: explicit Modal app name",
            )
            yield Static("[dim]App name preview: auto[/dim]", id="llama-app-preview")
            yield Static("")
            yield Button("Deploy", id="deploy-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self._selected_preset: str | None = None
        self._is_custom = False
        self._selected_gpu_type = DEFAULT_GPU_TYPE
        self._refresh_gpu_types()
        self._refresh_app_preview()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "preset-list":
            return
        opt_id = event.option.id or ""
        custom_fields = self.query_one("#custom-fields")
        if opt_id == "preset-custom":
            self._is_custom = True
            self._selected_preset = None
            custom_fields.remove_class("hidden")
        else:
            self._is_custom = False
            self._selected_preset = opt_id.removeprefix("preset-")
            custom_fields.add_class("hidden")
        self._refresh_app_preview()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "toggle-advanced":
            adv = self.query_one("#advanced-fields")
            adv.toggle_class("hidden")
        elif event.button.id == "deploy-btn":
            self._do_deploy()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id in {"repo-id", "instance-name-llama", "app-name-llama"}:
            self._refresh_app_preview()

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

    def action_do_deploy(self) -> None:
        self._do_deploy()

    def _do_deploy(self) -> None:
        config = DeploymentConfig(backend=BackendType.LLAMACPP)

        if self._is_custom:
            config.repo_id = self.query_one("#repo-id", Input).value.strip() or None
            config.quant = self.query_one("#quant", Input).value.strip() or None
            rev = self.query_one("#revision", Input).value.strip()
            config.revision = rev or None
        else:
            config.preset = self._selected_preset

        config.preload = self.query_one("#preload", Switch).value
        config.do_deploy = self.query_one("#do-deploy", Switch).value
        config.do_warmup = self.query_one("#warmup", Switch).value

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
        adv = self.query_one("#advanced-fields")
        if not adv.has_class("hidden"):
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

        model_hint = config.repo_id or config.preset
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

    def _refresh_app_preview(self) -> None:
        repo_id = self.query_one("#repo-id", Input).value.strip() if self._is_custom else ""
        model_hint = repo_id or self._selected_preset or "default"
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


class VllmDeployScreen(Screen):
    """vLLM deploy form."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+d", "do_deploy", "Deploy", show=True),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="menu-container"):
            yield Static("[bold cyan]Deploy vLLM[/bold cyan]  [dim]Step 2: Model & options[/dim]")
            yield Static("")

            yield Static("[bold]Model ranking[/bold]  [dim](Top 10 text-generation models)[/dim]")
            yield OptionList(
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
                default="Qwen/Qwen3-4B-Thinking-2507-FP8",
            )
            yield Static("GPU configuration", classes="form-label")
            with Horizontal(id="gpu-config-row-vllm"):
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
        self._rank_mode = "downloads"
        self._ranked_models: list[ModelCandidate] = []
        self._selected_gpu_type = DEFAULT_GPU_TYPE
        self._served_alias_touched = False
        self._updating_served_alias = False
        self._last_auto_served_alias = ""
        for widget in self.query(".vllm-advanced"):
            widget.add_class("hidden")
        self._refresh_gpu_types()
        self._sync_served_alias_from_model(force=True)
        self.app.begin_fetch_vllm_models(self._rank_mode, self)  # type: ignore[attr-defined]
        self._refresh_app_preview()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "vllm-rank-mode":
            selected_mode = "downloads" if event.option.id == "rank-downloads" else "trending"
            if selected_mode == self._rank_mode:
                return
            self._rank_mode = selected_mode
            self._ranked_models = []
            self._set_model_status("[dim]Loading model suggestions...[/dim]")
            self.query_one("#vllm-model-list", OptionList).set_options([])
            self.app.begin_fetch_vllm_models(self._rank_mode, self)  # type: ignore[attr-defined]
            return

        if event.option_list.id == "vllm-model-list":
            self._apply_ranked_model_selection(event.option.id or "")
            self._refresh_app_preview()
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

    def _set_model_status(self, text: str) -> None:
        self.query_one("#vllm-model-status", Static).update(text)

    def _apply_ranked_model_selection(self, option_id: str) -> None:
        if not option_id.startswith("model-"):
            return
        try:
            idx = int(option_id.split("-", 1)[1])
        except ValueError:
            return
        if idx < 0 or idx >= len(self._ranked_models):
            return
        self.query_one("#model-name", Input).value = self._ranked_models[idx].repo_id
        self._refresh_app_preview()

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
        n_gpu_str = self.query_one("#n-gpu", Input).value.strip()
        try:
            config.n_gpu = int(n_gpu_str) if n_gpu_str else 1
        except ValueError:
            config.n_gpu = 1
        adv_visible = not self.query(".vllm-advanced").first().has_class("hidden")
        if adv_visible:
            config.reasoning_parser = self.query_one("#reasoning-parser", Input).value.strip() or None
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
