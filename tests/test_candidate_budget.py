"""The candidate budget has to buy distinct models, not benchmark rows."""

from __future__ import annotations

import unittest

from unittest.mock import patch

import requests

from huggingface_hub.errors import HfHubHTTPError

from llm_launchpad.core.artificial_analysis import AAModelCandidate
from llm_launchpad.core.quick_deploy_refresh import (
    _aa_candidate_window,
    _deduped_candidates,
    _resolve_aa_model,
)
from llm_launchpad.protocol.models import CatalogExclusion


def _candidate(name: str, rank: int, parameter_count_b: float | None = None) -> AAModelCandidate:
    return AAModelCandidate(
        aa_model_id=f"id-{rank}", name=name, slug=name.lower().replace(" ", "-"),
        creator_name="", coding_score=None, intelligence_score=100.0 - rank,
        rank=rank, parameter_count_b=parameter_count_b, max_context_tokens=None,
    )


def _effort_variants(base: str, start_rank: int) -> list[AAModelCandidate]:
    return [
        _candidate(f"{base} ({effort})", start_rank + offset)
        for offset, effort in enumerate(("max", "xhigh", "high", "medium", "low"))
    ]


class DedupBeforeBudgetTests(unittest.TestCase):
    def test_effort_variants_collapse_to_their_best_row(self) -> None:
        deduped = _deduped_candidates(_effort_variants("Frontier API", 1))

        self.assertEqual(len(deduped), 1)
        # The first row is the highest-ranked, which is the one worth resolving.
        self.assertEqual(deduped[0].name, "Frontier API (max)")

    def test_a_variant_family_cannot_spend_the_whole_budget(self) -> None:
        # Sixteen API-only families at five effort settings each fills an
        # 80-row budget entirely, and the open-weight model behind them was
        # never inspected -- not rejected, just invisible.
        ranking: list[AAModelCandidate] = []
        for index in range(16):
            ranking.extend(_effort_variants(f"Frontier API {index}", 1 + index * 5))
        open_model = _candidate("Open Weights Model", 81)
        ranking.append(open_model)

        window = _aa_candidate_window(_deduped_candidates(ranking), 80)

        self.assertIn("Open Weights Model", {candidate.name for candidate in window})

    def test_the_budget_still_bounds_the_work(self) -> None:
        # Known-size candidates are bounded by their own bucket's budget.
        ranking = [
            _candidate(f"Model {index}", index + 1, parameter_count_b=27.0)
            for index in range(200)
        ]

        window = _aa_candidate_window(_deduped_candidates(ranking), 80)

        self.assertEqual(len(window), 80)

    def test_an_unstated_size_is_reachable_as_deep_as_any_bucket(self) -> None:
        # A candidate of unknown size could land in any bucket, so it must not
        # be reachable only a third as deep as one that states its size. Muse
        # Glimmer is a 30B open model with a GGUF repository whose feed row is
        # named "Muse Glimmer (high)"; under a single bucket-sized pool it sat
        # outside the window while models a hundred places below were checked.
        ranking = [_candidate(f"Model {index}", index + 1) for index in range(400)]

        window = _aa_candidate_window(_deduped_candidates(ranking), 80)

        self.assertEqual(len(window), 240)

    def test_the_pool_is_still_finite(self) -> None:
        # Depth is paid for with a request budget and remembered results, not
        # by removing the bound: an unbounded sweep is what exhausted the
        # Hugging Face quota and silently shortened the shortlist.
        ranking = [_candidate(f"Model {index}", index + 1) for index in range(2000)]

        window = _aa_candidate_window(_deduped_candidates(ranking), 80)

        self.assertEqual(len(window), 240)


class HubFailureVisibilityTests(unittest.TestCase):
    """A lookup that failed is not a model that has no weights."""

    def _candidate(self) -> AAModelCandidate:
        return _candidate("Some Open Model", 10)

    def test_rate_limiting_is_named_and_recorded(self) -> None:
        # Throttling removes whole stretches of the ranking at once. Reported
        # as nothing at all, a build that lost two thirds of a category to it
        # published as though it were complete.
        # Built through requests so the error carries a real 429 response;
        # the reason is read off response.status_code.
        response = requests.Response()
        response.status_code = 429
        exc = HfHubHTTPError("429 Too Many Requests", response=response)
        exclusions: list[CatalogExclusion] = []
        with patch(
            "llm_launchpad.core.quick_deploy_refresh._find_unsloth_gguf_match",
            side_effect=exc,
        ):
            resolved = _resolve_aa_model(
                self._candidate(), [], hf_api=object(), exclusions=exclusions
            )

        self.assertIsNone(resolved)
        self.assertEqual(len(exclusions), 1)
        self.assertIn("rate limit", exclusions[0].reason.lower())

    def test_any_other_lookup_failure_is_still_recorded(self) -> None:
        exclusions: list[CatalogExclusion] = []
        with patch(
            "llm_launchpad.core.quick_deploy_refresh._find_unsloth_gguf_match",
            side_effect=OSError("connection reset"),
        ):
            _resolve_aa_model(self._candidate(), [], hf_api=object(), exclusions=exclusions)

        self.assertEqual(len(exclusions), 1)
        self.assertNotIn("rate limit", exclusions[0].reason.lower())

    def test_a_model_with_genuinely_no_weights_stays_unlisted(self) -> None:
        # The benchmark feed is mostly API-only models. Listing every one of
        # them as "excluded" would bury the models that were really rejected.
        exclusions: list[CatalogExclusion] = []
        with patch(
            "llm_launchpad.core.quick_deploy_refresh._find_unsloth_gguf_match",
            return_value=None,
        ):
            _resolve_aa_model(self._candidate(), [], hf_api=object(), exclusions=exclusions)

        self.assertEqual(exclusions, [])
