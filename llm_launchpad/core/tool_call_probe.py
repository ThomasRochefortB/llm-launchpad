"""Verify that a ready endpoint can make an OpenAI-style tool call.

Coding agents drive a model entirely through tool calls: an endpoint that
chats but never returns ``tool_calls`` passes readiness and then does nothing
useful in OpenCode. The probe asks for one unambiguous call and checks that
the server parsed it into the structured field, which is what agents read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .shutdown import is_shutting_down

TOOL_CALLING_PASSED = "passed"
TOOL_CALLING_FAILED = "failed"

_PROBE_TOOL = "get_weather"
_PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": _PROBE_TOOL,
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
# Reasoning models think before calling; the budget has to cover that.
_PROBE_MAX_TOKENS = 4096
_PROBE_TIMEOUT = (10, 180)


@dataclass(frozen=True)
class ToolCallProbeResult:
    status: str
    detail: str

    @property
    def passed(self) -> bool:
        return self.status == TOOL_CALLING_PASSED


def tool_call_probe_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Call the {_PROBE_TOOL} tool for Paris. Do not answer in text.",
            }
        ],
        "tools": _PROBE_TOOLS,
        # "required" would let vLLM force a call through guided decoding even
        # when its tool parser is wrong, which is exactly what this checks.
        "tool_choice": "auto",
        "max_tokens": _PROBE_MAX_TOKENS,
        "temperature": 0,
    }


def classify_tool_call_response(payload: Any) -> ToolCallProbeResult:
    """Grade one chat-completions response body."""
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ToolCallProbeResult(TOOL_CALLING_FAILED, "response had no assistant message")
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not calls:
        content = str((message or {}).get("content") or "").strip()
        if _PROBE_TOOL in content:
            return ToolCallProbeResult(
                TOOL_CALLING_FAILED,
                "the model wrote the tool call as text; the server did not parse it",
            )
        return ToolCallProbeResult(
            TOOL_CALLING_FAILED, "the model answered in text instead of calling the tool"
        )
    function = (calls[0] or {}).get("function") or {}
    if function.get("name") != _PROBE_TOOL:
        return ToolCallProbeResult(
            TOOL_CALLING_FAILED, f"the model called an unknown tool {function.get('name')!r}"
        )
    arguments = function.get("arguments")
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except ValueError:
        return ToolCallProbeResult(TOOL_CALLING_FAILED, "tool arguments were not valid JSON")
    if not isinstance(parsed, dict):
        return ToolCallProbeResult(TOOL_CALLING_FAILED, "tool arguments were not a JSON object")
    return ToolCallProbeResult(TOOL_CALLING_PASSED, "tool call parsed")


def verify_tool_calling(
    url: str,
    model: str | None,
    api_key: str | None,
    *,
    attempts: int = 2,
) -> ToolCallProbeResult | None:
    """Probe tool calling; ``None`` means the probe was cancelled."""
    import requests

    root = url.rstrip("/").removesuffix("/v1")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    result = ToolCallProbeResult(TOOL_CALLING_FAILED, "not attempted")
    try:
        if not model:
            response = requests.get(f"{root}/v1/models", headers=headers, timeout=(10, 30))
            response.raise_for_status()
            model = response.json()["data"][0]["id"]
        for _ in range(max(1, attempts)):
            if is_shutting_down():
                return None
            response = requests.post(
                f"{root}/v1/chat/completions",
                headers=headers,
                json=tool_call_probe_payload(str(model)),
                timeout=_PROBE_TIMEOUT,
            )
            if response.status_code >= 400:
                body = (response.text or "").strip().replace("\n", " ")
                # A rejected request is deterministic; asking again won't help.
                return ToolCallProbeResult(
                    TOOL_CALLING_FAILED, f"HTTP {response.status_code}: {body[:200]}"
                )
            result = classify_tool_call_response(response.json())
            if result.passed:
                return result
    except Exception as exc:
        return ToolCallProbeResult(TOOL_CALLING_FAILED, f"request failed: {exc}")
    return result
