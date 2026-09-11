"""Vast REST account validation, discovery, and instance lifecycle."""

from __future__ import annotations

import math
import json
import os
from typing import Any

import requests

from ..protocol.models import OfferCostBreakdown, VastAuthStatus, VastInstance, VastOffer, VastOfferQuery
from .coerce import optional_float, positive_int
from .vast_auth import VastCredentials, normalize_vast_api_key, resolve_vast_credentials

VAST_API_URL = "https://console.vast.ai/api/v0"


class VastApiError(RuntimeError):
    """A sanitized provider error suitable for CLI/TUI presentation."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _nonnegative(value: Any) -> float | None:
    try:
        parsed = optional_float(value)
    except OverflowError:
        return None
    return parsed if parsed is not None and math.isfinite(parsed) and parsed >= 0 else None


def _text(value: Any) -> str:
    # Provider strings also reach terminal output: remove control characters.
    return "".join(char for char in value if char.isprintable()).strip() if isinstance(value, str) else ""


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "True"):
        return True
    if value in (0, "0", "false", "False"):
        return False
    return None


def vast_offer_search_payload(query: VastOfferQuery) -> dict[str, Any]:
    """Compile explicit on-demand filters and the requested storage allocation."""
    if (query.gpu_count is not None and positive_int(query.gpu_count) is None) or positive_int(query.disk_gb) is None:
        raise ValueError("GPU count and disk size must be positive integers.")
    reliability = _nonnegative(query.min_reliability)
    if reliability is None or reliability > 1:
        raise ValueError("Minimum reliability must be between 0 and 1.")
    if positive_int(query.limit) is None or query.limit > 500:
        raise ValueError("Offer limit must be between 1 and 500.")
    payload: dict[str, Any] = {
        "type": "on-demand", "verified": {"eq": True},
        "rentable": {"eq": True}, "rented": {"eq": False},
        "gpu_arch": {"eq": "nvidia"}, "cpu_arch": {"eq": "amd64"},
        "num_gpus": {"eq": query.gpu_count} if query.gpu_count is not None else {"gte": 1, "lte": 8},
        "disk_space": {"gte": query.disk_gb},
        "reliability": {"gte": reliability},
        "allocated_storage": query.disk_gb, "limit": query.limit,
        "order": [["dph_total", "asc"]],
    }
    if query.gpu_type and query.gpu_type.strip():
        payload["gpu_name"] = {"eq": query.gpu_type.strip().replace("_", " ")}
    if query.country:
        country = query.country.strip().upper()
        if len(country) != 2 or not country.isascii() or not country.isalpha():
            raise ValueError("Vast --region must be a two-letter country code, such as US.")
        payload["geolocation"] = {"eq": country}
    if query.datacenter_only:
        payload["datacenter"] = {"eq": True}
    return payload


def _compute_capability(value: Any) -> float | None:
    """Convert Vast's packed compute capability (860) to its real form (8.6)."""
    packed = _nonnegative(value)
    if packed is None or packed <= 0:
        return None
    return round(packed / 100.0, 1)


def parse_vast_offer(raw: Any, query: VastOfferQuery) -> VastOffer | None:
    """Normalize known rental fields and reject invalid or ineligible rows."""
    if not isinstance(raw, dict):
        return None
    offer_id, machine_id = positive_int(raw.get("id")), positive_int(raw.get("machine_id"))
    count = positive_int(raw.get("num_gpus"))
    memory = _nonnegative(raw.get("gpu_ram"))
    reliability = _nonnegative(raw.get("reliability"))
    disk = _nonnegative(raw.get("disk_space"))
    verified = _boolean(raw.get("verified"))
    if verified is None:
        verified = raw.get("verification") == "verified"
    gpu_type = _text(raw.get("gpu_name"))
    if (
        not offer_id or not machine_id or not gpu_type or count is None
        or (query.gpu_count is not None and count != query.gpu_count)
        or (query.gpu_count is None and count is not None and count > 8)
        or memory is None or memory <= 0
        or reliability is None or not query.min_reliability <= reliability <= 1
        or disk is None or disk < query.disk_gb
        or not verified or _boolean(raw.get("rentable")) is not True
        # Vast's CLI treats a null rented field as available.
        or (raw.get("rented") is not None and _boolean(raw["rented"]) is not False)
        or _boolean(raw.get("is_bid", False)) is not False
        or _text(raw.get("gpu_arch")).lower() != "nvidia"
        or _text(raw.get("cpu_arch")).lower() not in {"amd64", "x86_64"}
    ):
        return None
    datacenter = _boolean(raw.get("datacenter")) is True
    if query.datacenter_only and not datacenter:
        return None
    if query.gpu_type and gpu_type.casefold().replace("_", " ") != query.gpu_type.strip().casefold().replace("_", " "):
        return None
    # geolocation may contain a city as well as a country; the API applies the
    # country filter. Preserve it as a display label, not an ISO country field.
    search = raw.get("search")
    pricing = search if isinstance(search, dict) else {}
    compute = _nonnegative(pricing.get("gpuCostPerHour", raw.get("dph_base")))
    disk_hour = _nonnegative(pricing.get("diskHour"))
    total = _nonnegative(raw.get("dph_total"))
    if disk_hour is None and total is not None and compute is not None and total >= compute:
        disk_hour = total - compute
    if total is None and compute is not None and disk_hour is not None:
        total = compute + disk_hour
    cpu_ram = _nonnegative(raw.get("cpu_ram"))
    duration = _nonnegative(raw.get("duration"))
    if duration is not None and duration == 0:
        return None
    return VastOffer(
        id=str(offer_id), machine_id=str(machine_id), gpu_type=gpu_type,
        gpu_count=count, gpu_memory_gb=memory / 1000,
        reliability=reliability, disk_gb=query.disk_gb,
        disk_capacity_gb=disk,
        cuda_max_good=_nonnegative(raw.get("cuda_max_good")),
        compute_capability=_compute_capability(raw.get("compute_cap")),
        inet_down_mbps=_nonnegative(raw.get("inet_down")),
        location=_text(raw.get("geolocation")), datacenter=datacenter,
        cpu_memory_gb=cpu_ram / 1000 if cpu_ram is not None else None,
        max_duration_hours=duration / 3600 if duration is not None else None,
        costs=OfferCostBreakdown(
            compute_per_hour_usd=compute, disk_per_hour_usd=disk_hour,
            total_per_hour_usd=total,
            download_per_gb_usd=_nonnegative(raw.get("inet_down_cost")),
            upload_per_gb_usd=_nonnegative(raw.get("inet_up_cost")),
        ),
    )


class VastBackend:
    """REST API client; discovery never allocates billable resources."""

    def __init__(self, credentials: VastCredentials | None = None) -> None:
        self.credentials = credentials if credentials is not None else resolve_vast_credentials()

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None,
        *, version: int = 0, params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.credentials.api_key:
            raise VastApiError("No Vast API key configured. Run llm-launchpad vast-auth login.")
        key = normalize_vast_api_key(self.credentials.api_key)
        try:
            response = requests.request(
                method, f"{VAST_API_URL.removesuffix('0')}{version}{path}", json=payload, params=params,
                headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                timeout=(5, 20), allow_redirects=False,
            )
        except requests.RequestException:
            raise VastApiError("Could not reach Vast. Check your connection and retry.") from None
        try:
            if response.status_code in {401, 403}:
                raise VastApiError("Vast rejected the API key or its permissions.")
            if response.status_code == 429:
                raise VastApiError(
                    "Vast rate limit reached. Wait briefly and retry.", status_code=429
                )
            if not 200 <= response.status_code < 300:
                # Vast explains its rejections in the body, but a response can
                # echo the request back -- environment and startup script
                # included -- so it stays out of the message unless someone
                # explicitly asks to see it.
                body = response.text
                detail = ""
                if os.environ.get("LLM_LAUNCHPAD_API_DEBUG") == "1" and isinstance(body, str):
                    reason = " ".join(body.split())[:200]
                    detail = f" {reason}" if reason else ""
                raise VastApiError(
                    f"Vast request failed (HTTP {response.status_code}).{detail}",
                    status_code=response.status_code,
                )
            try:
                data = response.json()
            except ValueError:
                raise VastApiError("Vast returned invalid JSON.") from None
            if not isinstance(data, dict) or data.get("success") is False or data.get("error"):
                raise VastApiError("Vast returned an unsuccessful or malformed response.")
            return data
        finally:
            response.close()

    def account_id(self) -> str:
        """Read the account identity, preserving errors for lifecycle retries."""
        data = self._request("GET", "/users/current/")
        account_id = positive_int(data.get("id"))
        if account_id is None:
            raise VastApiError("Vast account response is missing a valid account ID.")
        return str(account_id)

    def auth_status(self) -> VastAuthStatus:
        """Validate the effective credential without exposing account secrets."""
        try:
            account_id = self.account_id()
        except (VastApiError, ValueError) as exc:
            return VastAuthStatus(False, source=self.credentials.source, error=str(exc))
        return VastAuthStatus(True, source=self.credentials.source, account_id=account_id)

    def list_offers(self, query: VastOfferQuery | None = None) -> list[VastOffer]:
        """Fetch a bounded preview of eligible offers, with unknown prices last."""
        query = query or VastOfferQuery()
        payload = self._request("POST", "/bundles/", vast_offer_search_payload(query))
        raw = payload.get("offers")
        if not isinstance(raw, list):
            raise VastApiError("Vast offer response must contain an offers list.")
        offers: dict[str, VastOffer] = {}
        for row in raw:
            offer = parse_vast_offer(row, query)
            if offer is not None:
                offers[offer.id] = offer
        return sorted(offers.values(), key=lambda row: (
            row.costs.total_per_hour_usd is None,
            row.costs.total_per_hour_usd if row.costs.total_per_hour_usd is not None else math.inf,
            row.id,
        ))[:query.limit]

    def get_offer(self, offer_id: str, query: VastOfferQuery) -> VastOffer:
        """Reprice one selected offer with the requested disk before rental."""
        offer_id = _resource_id(offer_id)
        payload = vast_offer_search_payload(query)
        # Search's `id` addresses the bundle, while returned `id` is the
        # rentable ask contract. Filtering `id` wrongly hides available asks.
        payload["ask_contract_id"] = {"eq": int(offer_id)}
        data = self._request("POST", "/bundles/", payload)
        rows = data.get("offers")
        if isinstance(rows, list):
            for raw in rows:
                offer = parse_vast_offer(raw, query)
                if offer is not None and offer.id == offer_id:
                    return offer
        raise VastApiError("The selected Vast offer is no longer available with these requirements.")

    def create_instance(
        self, offer_id: str, *, image: str, disk_gb: int, label: str, onstart: str = "",
    ) -> str:
        """Accept an offer once. Never automatically retry this billable request."""
        payload = {
            "client_id": "me", "image": image, "disk": disk_gb,
            "label": label, "runtype": "ssh", "target_state": "running",
            "cancel_unavail": True,
        }
        if onstart:
            payload["onstart"] = onstart
        data = self._request("PUT", f"/asks/{_resource_id(offer_id)}/", payload)
        instance_id = positive_int(data.get("new_contract"))
        if data.get("success") is not True or instance_id is None:
            raise VastApiError("Vast rental result is uncertain; reconcile the recorded label before retrying.")
        return str(instance_id)

    def get_instance(self, instance_id: str) -> VastInstance | None:
        """Read a rental; 404 or an explicit null instance establishes absence."""
        try:
            data = self._request("GET", f"/instances/{_resource_id(instance_id)}/")
        except VastApiError as exc:
            if exc.status_code == 404:
                return None
            raise
        if "instances" in data and data["instances"] is None:
            return None
        row = _parse_instance(data.get("instances"))
        if row.id != instance_id:
            raise VastApiError("Vast returned an unexpected instance identity.")
        return row

    def find_instances(self, label: str) -> list[VastInstance]:
        """Reconcile an uncertain create by its unique label, across every page."""
        rows: list[VastInstance] = []
        token: str | None = None
        seen: set[str] = set()
        for _ in range(100):
            params = {"select_filters": json.dumps({"label": {"eq": label}}), "limit": 25}
            if token:
                params["after_token"] = token
            data = self._request("GET", "/instances/", version=1, params=params)
            raw = data.get("instances")
            if not isinstance(raw, list):
                raise VastApiError("Vast returned a malformed instance list.")
            for item in raw:
                row = _parse_instance(item)
                if row.label == label:
                    rows.append(row)
            token = data.get("next_token")
            if not token:
                return rows
            if not isinstance(token, str) or token in seen:
                break
            seen.add(token)
        raise VastApiError("Vast instance pagination did not complete; rental state remains uncertain.")

    def attach_key(self, instance_id: str, public_key: str) -> None:
        """Authorize one instance and require explicit confirmation from Vast."""
        data = self._request("POST", f"/instances/{_resource_id(instance_id)}/ssh/", {"ssh_key": public_key})
        if data.get("success") is not True:
            raise VastApiError("Vast did not confirm SSH key attachment.")

    def destroy_instance(self, instance_id: str) -> None:
        """Destroy a rental and its disk; callers confirm removal before forgetting it."""
        try:
            data = self._request("DELETE", f"/instances/{_resource_id(instance_id)}/")
        except VastApiError as exc:
            if exc.status_code == 404:
                return
            raise
        if data.get("success") is not True:
            raise VastApiError("Vast did not confirm instance destruction.")


def _resource_id(value: str) -> str:
    if not value.isascii() or not value.isdigit() or positive_int(value) is None:
        raise ValueError("A positive numeric Vast resource ID is required.")
    return str(int(value))


def _parse_instance(raw: Any) -> VastInstance:
    if not isinstance(raw, dict) or positive_int(raw.get("id")) is None:
        raise VastApiError("Vast returned malformed instance metadata.")
    return VastInstance(
        id=str(raw["id"]), label=_text(raw.get("label")),
        state=_text(raw.get("actual_status")), machine_id=str(raw.get("machine_id") or ""),
        ssh_host=_text(raw.get("ssh_host")), ssh_port=positive_int(raw.get("ssh_port")) or 0,
        status_msg=_text(raw.get("status_msg")),
    )
