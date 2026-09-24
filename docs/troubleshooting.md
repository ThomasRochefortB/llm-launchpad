# Troubleshooting

Run `llm-launchpad doctor` first: it checks the Modal CLI, Modal, Prime
Intellect, and Hugging Face authentication, the optional Artificial Analysis
key, and local state-directory writability, then prints a fix hint for each
failure.

## Common issues

- `Modal CLI not found`: reinstall or upgrade the package, then confirm `modal --help` works in the same shell.
- `Modal authentication missing`: run `modal setup`.
- Hugging Face download errors: run `hf auth login` and verify the model license or gated-repo access in your Hugging Face account.
- Warmup stays queued: Modal may still be scheduling the requested GPU. Try a smaller GPU configuration or wait for capacity.
- Endpoint status fails after deploy: inspect `llm-launchpad logs --backend <backend> --instance-name <name>` for backend startup errors.
- First llama.cpp deploy after a large download dies at Modal's web-server startup timeout: the GPU container now sequentially reads GGUF shards (and a projector, if any) before `llama-server` starts, and the bind wait defaults to 90 minutes. Raise `LLAMACPP_SERVE_STARTUP_TIMEOUT_MINUTES` if a still-larger first read needs more time, or set `LLAMACPP_WARM_VOLUME=false` only to skip hydration.
- Prime pod cannot be managed: Launchpad's local `ssh`/`ssh-keygen` binaries must be available; the bootstrap key lives under `~/.llm_launchpad/prime/`.
- Settings look reset: `settings.json` may be corrupt; the TUI Settings screen shows load diagnostics.

## Debug log

Launchpad writes a rotating debug log (a few MiB, three rotated backups) to:

```
~/.llm_launchpad/logs/llm_launchpad.log
```

It records background failure paths that are intentionally silent on screen,
such as cache persistence, auth probing, and subprocess cleanup, so include a
redacted tail when filing an issue. Secrets such as bearer keys and tokens are
never written to the log.

## TUI specifics

- The TUI uses keyboard navigation: Tab/Shift+Tab moves between controls, arrows navigate lists, and Enter activates the focused control. Mouse selection belongs to your terminal; there is no mouse-mode toggle.
- Copy selected text with your terminal's Copy shortcut, then focus an input and use the terminal's Paste shortcut (often Ctrl+Shift+C/V on Linux or Cmd+C/V on macOS). Terminal paste works over SSH without reading the remote machine's clipboard. Plain Ctrl+C is the app's copy-selection/quit key, not the terminal Copy shortcut.
- If tmux captures dragging, use your terminal's selection override (usually Shift+drag, or Option+drag in iTerm2), then its Copy shortcut. This bypasses tmux clipboard forwarding.
- App Copy buttons and `y` use the local clipboard or OSC 52 terminal requests. Remote clipboard writes cannot be confirmed and require terminal support; in tmux, `set -g allow-passthrough on` enables the app's passthrough request. Terminal-native selection is the fallback when clipboard requests are blocked. Plain Ctrl+V can paste from the host clipboard or the last text copied within the app; use the terminal Paste shortcut for your local clipboard over SSH.

## Filing an issue

Open a GitHub issue using the bug report template and include:

1. `llm-launchpad doctor` output.
2. A redacted tail of `~/.llm_launchpad/logs/llm_launchpad.log`.
3. The command or TUI flow that failed, plus the installed version (`llm-launchpad --version`).
