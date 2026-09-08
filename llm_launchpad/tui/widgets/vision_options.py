"""Shared image input controls for manual and curated deployments."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Select, Static

from ...protocol.enums import BackendType, VisionMode
from ...protocol.models import DeploymentConfig
from ...core.vision import validate_vision_options


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
        if self.backend == BackendType.LLAMACPP:
            yield Input(placeholder="Projector repository override (optional)", id="projector-repo")
            yield Input(placeholder="Projector revision override (optional)", id="projector-revision")
            yield Input(placeholder="Exact projector filename (optional)", id="projector-file")
        elif self.backend == BackendType.VLLM:
            yield Static("Maximum images per prompt", classes="form-label")
            yield Input(value="1", type="integer", id="image-limit")
            yield Input(placeholder="Image processor kwargs JSON (optional)", id="mm-processor-kwargs")

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
