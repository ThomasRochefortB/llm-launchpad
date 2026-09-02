# OpenCode integration

LLM-Launchpad automatically detects a local installation of
[OpenCode](https://github.com/anomalyco/opencode) and syncs Launchpad-managed
deployments into your OpenCode config with the final OpenAI-compatible base
URL and model ID after deployment.

- Stale provider entries are pruned when apps stop or disappear.
- Sync runs automatically after `list` and `stop`, or on demand:

```bash
llm-launchpad opencode sync --dry-run
```

Provider naming is aligned with active deployments so synced connections stay
clean and predictable.
