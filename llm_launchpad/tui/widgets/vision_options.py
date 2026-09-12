"""Shared image input controls for manual and curated deployments."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Select, Static

from ...protocol.enums import BackendType, VisionMode
from ...protocol.models import DeploymentConfig
from ...core.vision import validate_vision_options
from .input_form import FormField


class VisionOptions(Vertical):
    """Keep vision settings consistent across deploy forms."""

    DEFAULT_CSS = "VisionOptions { height: auto; margin-bottom: 1; }"

    def __init__(self, backend: BackendType | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.backend = backend

    def compose(self) -> ComposeResult:
        yield Static("Image input", classes="form-label")
        yield Select(
            [("Automatic vision", "auto"), ("Require vision", "on"), ("Text only", "off")],
            value="auto", allow_blank=False, id="vision-mode",
        )
        # Everything below only describes how images are handled, so it is
        # hidden outright once the deployment is text-only. Each field carries
        # a real label: a placeholder disappears the moment it is typed into.
        with Vertical(id="vision-detail-fields"):
            if self.backend == BackendType.LLAMACPP:
                yield FormField(
                    "Projector repository (optional)",
                    "projector-repo",
                    hint="Overrides the mmproj repo auto-detected from the model",
                )
                yield FormField(
                    "Projector revision (optional)",
                    "projector-revision",
                    hint="Leave blank to use the projector repo's default branch",
                )
                yield FormField(
                    "Projector filename (optional)",
                    "projector-file",
                    hint="e.g., mmproj-model-f16.gguf",
                )
            elif self.backend == BackendType.VLLM:
                yield FormField(
                    "Maximum images per prompt",
                    "image-limit",
                    default="1",
                    input_type="integer",
                    hint="vLLM --limit-mm-per-prompt image",
                )
                yield FormField(
                    "Image processor kwargs (JSON, optional)",
                    "mm-processor-kwargs",
                    hint='e.g., {"max_pixels": 1003520}',
                )

    def on_mount(self) -> None:
        self._sync_detail_visibility()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "vision-mode":
            self._sync_detail_visibility()

    def _sync_detail_visibility(self) -> None:
        try:
            detail = self.query_one("#vision-detail-fields", Vertical)
        except Exception:
            return
        mode = self.query_one("#vision-mode", Select).value
        detail.set_class(mode == VisionMode.OFF.value, "hidden")

    def apply(self, config: DeploymentConfig) -> None:
        """Copy and validate form values before beginning deployment."""
        config.vision_mode = VisionMode(self.query_one("#vision-mode", Select).value)
        if self.backend == BackendType.LLAMACPP:
            config.projector_repo = self.query_one("#projector-repo", Input).value.strip() or None
            config.projector_revision = self.query_one("#projector-revision", Input).value.strip() or None
            config.projector_file = self.query_one("#projector-file", Input).value.strip() or None
        elif self.backend == BackendType.VLLM:
            try:
                config.image_limit = int(self.query_one("#image-limit", Input).value)
            except ValueError:
                raise ValueError("Image limit must be an integer of at least one.") from None
            config.mm_processor_kwargs = self.query_one("#mm-processor-kwargs", Input).value.strip() or None
        validate_vision_options(config)
