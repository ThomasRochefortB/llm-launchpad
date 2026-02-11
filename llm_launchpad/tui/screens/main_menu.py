"""Main menu screen: deploy, manage endpoints, settings.

Inspired by the Codex TUI main screen with a prominent banner,
auth status line, and keyboard-navigable option list.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Footer, OptionList, Static
from textual.widgets.option_list import Option

BANNER = r"""[bold cyan]
 _     _     __  __
| |   | |   |  \/  |
| |   | |   | |\/| |
| |___| |___| |  | |
|_____|_____|_|  |_|
    LAUNCHPAD
[/bold cyan]"""


class MainMenuScreen(Screen):
    """Top-level menu: deploy, manage, settings."""

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "select_deploy", "Deploy", show=True),
        Binding("m", "select_manage", "Manage", show=True),
        Binding("s", "select_settings", "Settings", show=True),
    ]

    def __init__(self, username: str = "", version: str = "") -> None:
        super().__init__()
        self.username = username
        self.version = version

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="menu-container"):
                yield Static(BANNER, id="banner-text")
                version_text = f"v{self.version}  " if self.version else ""
                yield Static(
                    f"[bold]{version_text}[/bold][dim]Modal LLM backends[/dim]",
                    classes="centered",
                )
                yield Static("")  # spacer
                if self.username:
                    yield Static(
                        f"[green]  Authenticated as {self.username}[/green]",
                        id="auth-status",
                    )
                yield Static("")  # spacer
                yield Static("[bold]Choose action[/bold]  [dim](use arrow keys, enter to select)[/dim]")
                yield OptionList(
                    Option("  Deploy                  Launch a new LLM backend", id="deploy"),
                    Option("  Manage endpoints        List, status, logs, stop", id="manage"),
                    Option("  Settings                GPU config, scaledown", id="settings"),
                    id="action-list",
                )
        yield Footer()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id
        if option_id == "deploy":
            self.app.action_push_deploy()  # type: ignore[attr-defined]
        elif option_id == "manage":
            self.app.action_push_manage()  # type: ignore[attr-defined]
        elif option_id == "settings":
            self.app.action_push_settings()  # type: ignore[attr-defined]

    def action_select_deploy(self) -> None:
        self.app.action_push_deploy()  # type: ignore[attr-defined]

    def action_select_manage(self) -> None:
        self.app.action_push_manage()  # type: ignore[attr-defined]

    def action_select_settings(self) -> None:
        self.app.action_push_settings()  # type: ignore[attr-defined]

    def action_quit(self) -> None:
        self.app.exit()
