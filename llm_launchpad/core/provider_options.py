"""Typed accessors for provider-specific deployment settings."""

from __future__ import annotations

from ..protocol.models import DeploymentConfig, PrimeProviderOptions


def prime_provider_options(config: DeploymentConfig) -> PrimeProviderOptions:
    """Return Prime options or reject options intended for another provider."""

    options = config.provider_options
    if options is None:
        return PrimeProviderOptions()
    if not isinstance(options, PrimeProviderOptions):
        raise ValueError("Deployment config contains non-Prime provider options.")
    return options
