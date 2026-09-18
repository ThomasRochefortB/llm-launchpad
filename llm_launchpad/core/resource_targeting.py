"""Shared resource targeting: what a cancellation or cleanup can stop.

Pure helper owned by neither the job worker nor the lifecycle runner. Both
sides supply what they know (provider, app name, observed resource id, whether
anything could have been allocated); this module decides whether a stop is
possible and what handle to use.

Addressing rules:
- Modal apps are addressed by name (``modal app stop <name>``).
- Vast rentals resolve from the record persisted under the deployment name.
- Prime termination genuinely needs a pod id.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..protocol.enums import ComputeProvider


NAME_ADDRESSABLE_PROVIDERS = frozenset({ComputeProvider.MODAL, ComputeProvider.VAST})


@dataclass(frozen=True)
class StopTarget:
    """Resolved stop target for one resource."""

    resource_id: str | None
    stoppable: bool
    reason: str = ""


def resolve_stop_target(
    *,
    provider: ComputeProvider,
    app_name: str | None,
    resource_id: str | None,
    allocated: bool,
) -> StopTarget:
    """Decide what a cancellation/cleanup can stop.

    ``resource_id`` is the observed allocation handle (pod id, instance id,
    ...). ``allocated`` records whether the caller believes provisioning may
    already have happened server-side even when no id event arrived (e.g. a
    name-addressable app may exist under its name).
    """
    identifier = (resource_id or "").strip()
    if identifier:
        return StopTarget(resource_id=identifier, stoppable=True)
    if not allocated:
        return StopTarget(
            resource_id=None, stoppable=False, reason="nothing allocated"
        )
    name = (app_name or "").strip()
    if provider in NAME_ADDRESSABLE_PROVIDERS and name:
        return StopTarget(resource_id=None, stoppable=True, reason="by-name")
    return StopTarget(
        resource_id=None, stoppable=False, reason="no addressable handle"
    )


__all__ = ["NAME_ADDRESSABLE_PROVIDERS", "StopTarget", "resolve_stop_target"]
