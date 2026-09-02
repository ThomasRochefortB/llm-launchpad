# Troubleshooting

Run `llm-launchpad doctor` first: it checks the Modal CLI, Modal, Prime
Intellect, and Hugging Face authentication, the optional Artificial Analysis
key, and local state-directory writability, then prints a fix hint for each
failure.

## Common issues

- `Modal CLI not found`: reinstall or upgrade the package, then confirm `modal --help` works in the same shell.
- `Modal authentication missing`: run `modal setup`.
- Hugging Face download errors: run `huggingface-cli login` and verify the model license or gated-repo access in your Hugging Face account.
- Warmup stays queued: Modal may still be scheduling the requested GPU. Try a smaller GPU configuration or wait for capacity.
- Endpoint status fails after deploy: inspect `llm-launchpad logs --backend <backend> --instance-name <name>` for backend startup errors.
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

- TUI copy/paste: select text by dragging in mouse mode, then use `Ctrl/Cmd+C`; `Ctrl/Cmd+V` pastes the host clipboard into focused fields.
- Over SSH, start with `llm-launchpad tui --no-mouse` to let the terminal handle native text selection.

## Filing an issue

Open a GitHub issue using the bug report template and include:

1. `llm-launchpad doctor` output.
2. A redacted tail of `~/.llm_launchpad/logs/llm_launchpad.log`.
3. The command or TUI flow that failed, plus the installed version (`llm-launchpad --version`).
