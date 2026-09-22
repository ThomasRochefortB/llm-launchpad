"""Tool-call probe grading and its effect on published connections."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_launchpad.core.opencode import (
    OpenCodeConnection,
    _provider_payload,
    build_openai_connection_payload,
)
from llm_launchpad.core.tool_call_probe import (
    TOOL_CALLING_FAILED,
    TOOL_CALLING_PASSED,
    classify_tool_call_response,
)
# Bound at import, before conftest's autouse stub replaces the module attribute.
from llm_launchpad.core.tool_call_probe import verify_tool_calling as real_verify_tool_calling
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import DeploymentConfig


def _response(message: dict) -> dict:
    return {"choices": [{"message": message}]}


def test_parsed_tool_call_passes() -> None:
    result = classify_tool_call_response(
        _response(
            {
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
                ]
            }
        )
    )
    assert result.passed


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ({"content": "It is sunny."}, "answered in text"),
        ({"content": '<tool_call>{"name": "get_weather"}</tool_call>'}, "did not parse"),
        (
            {"tool_calls": [{"function": {"name": "get_weather", "arguments": "{bad"}}]},
            "not valid JSON",
        ),
        ({"tool_calls": [{"function": {"name": "other", "arguments": "{}"}}]}, "unknown tool"),
    ],
)
def test_unusable_responses_fail_with_reason(message: dict, reason: str) -> None:
    result = classify_tool_call_response(_response(message))
    assert result.status == TOOL_CALLING_FAILED
    assert reason in result.detail


def test_http_rejection_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def post(url, headers, json, timeout):  # noqa: A002 - mirrors requests
        calls.append(json)
        return SimpleNamespace(status_code=400, text='"auto" tool choice requires --enable-auto-tool-choice')

    import requests

    monkeypatch.setattr(requests, "post", post)
    outcome = real_verify_tool_calling("https://example.test/v1", "m", "key")
    assert outcome is not None
    assert outcome.status == TOOL_CALLING_FAILED
    assert "enable-auto-tool-choice" in outcome.detail
    assert len(calls) == 1
    assert calls[0]["tool_choice"] == "auto"


def test_failed_tool_calling_reaches_opencode_config() -> None:
    config = DeploymentConfig(
        backend=BackendType.VLLM,
        model_name="Qwen/Qwen3-4B",
        app_name="qwen",
        tool_calling=TOOL_CALLING_FAILED,
    )
    payload = build_openai_connection_payload(config, "https://example.test")
    assert payload["tool_calling"] == TOOL_CALLING_FAILED
    connection = OpenCodeConnection(
        app_name="qwen",
        instance_name="qwen",
        provider_id="p",
        provider_name="n",
        base_url="https://example.test/v1",
        model_id="qwen",
        display_name="Qwen",
        backend=BackendType.VLLM,
        tool_calling=TOOL_CALLING_FAILED,
    )
    model = _provider_payload(connection)["models"]["qwen"]
    assert model["tool_call"] is False
    passed = OpenCodeConnection(**{**connection.__dict__, "tool_calling": TOOL_CALLING_PASSED})
    assert "tool_call" not in _provider_payload(passed)["models"]["qwen"]
    json.dumps(model)
