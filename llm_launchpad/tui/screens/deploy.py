"""Deploy screen: backend selection, model config, deploy options.

Two sub-flows: llama.cpp (preset/custom GGUF) and vLLM (model params).
Keyboard-driven form navigation with enter-to-proceed.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Input,
    OptionList,
    Static,
    Switch,
)
from textual.widgets.option_list import Option

from ...core.hf_models import ModelCandidate
from ...protocol.enums import BackendType
from ...protocol.models import DeploymentConfig
from ...presets import PRESETS
from ..workers import VllmModelsFailed, VllmModelsLoaded
from ..widgets.input_form import FormField, ToggleField


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
            yield Button("Deploy", id="deploy-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self._selected_preset: str | None = None
        self._is_custom = False

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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "toggle-advanced":
            adv = self.query_one("#advanced-fields")
            adv.toggle_class("hidden")
        elif event.button.id == "deploy-btn":
            self._do_deploy()

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

        self.app.begin_deploy(config)  # type: ignore[attr-defined]

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
            yield FormField(
                "Model revision (optional)",
                "model-revision",
                hint="Leave blank to use default branch",
            )
            yield FormField(
                "Served model alias",
                "served-model-name",
                default="llm",
            )
            yield ToggleField("Fast boot (enforce eager)", "fast-boot", default=True)
            yield FormField(
                "Number of GPUs (tensor parallel)",
                "n-gpu",
                default="1",
            )

            yield Static("")
            yield ToggleField("Deploy the server", "do-deploy-vllm", default=True)
            yield ToggleField("Run smoke test (if not deploying)", "run-smoke", default=False)
            yield ToggleField("Warm up after deploy", "warmup-vllm", default=True)

            yield Static("")
            yield Button("Deploy", id="deploy-vllm-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self._rank_mode = "downloads"
        self._ranked_models: list[ModelCandidate] = []
        self.app.begin_fetch_vllm_models(self._rank_mode, self)  # type: ignore[attr-defined]

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

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Mirror highlighted model into the input for keyboard-first flow."""
        if event.option_list.id != "vllm-model-list":
            return
        self._apply_ranked_model_selection(event.option.id or "")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "deploy-vllm-btn":
            self._do_deploy()

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

    def _do_deploy(self) -> None:
        config = DeploymentConfig(backend=BackendType.VLLM)
        config.model_name = self.query_one("#model-name", Input).value.strip() or None
        config.model_revision = self.query_one("#model-revision", Input).value.strip() or None
        config.served_model_name = self.query_one("#served-model-name", Input).value.strip() or None
        config.fast_boot = self.query_one("#fast-boot", Switch).value
        n_gpu_str = self.query_one("#n-gpu", Input).value.strip()
        try:
            config.n_gpu = int(n_gpu_str) if n_gpu_str else 1
        except ValueError:
            config.n_gpu = 1
        config.do_deploy = self.query_one("#do-deploy-vllm", Switch).value
        config.run_smoke = self.query_one("#run-smoke", Switch).value
        config.do_warmup = self.query_one("#warmup-vllm", Switch).value

        self.app.begin_deploy(config)  # type: ignore[attr-defined]

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
