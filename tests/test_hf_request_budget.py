"""Discovery must know what it is spending, and remember what it learned."""

from __future__ import annotations

from datetime import datetime, timedelta, UTC
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_launchpad.core.artificial_analysis import AAModelCandidate
from llm_launchpad.core.hf_budget import (
    DEFAULT_BUDGET,
    HF_REQUESTS_PER_WINDOW,
    HubBudgetExhausted,
    HubRequestBudget,
    UnlimitedBudget,
)
from llm_launchpad.core.hf_repo_matches import (
    HIT_TTL,
    MISS_TTL,
    RepoMatchStore,
)
from llm_launchpad.core.quick_deploy_refresh import _resolve_aa_model
from llm_launchpad.protocol.models import CatalogExclusion


def _candidate(name: str = "Some Open Model") -> AAModelCandidate:
    return AAModelCandidate(
        aa_model_id="id", name=name, slug=name.lower().replace(" ", "-"),
        creator_name="", coding_score=None, intelligence_score=20.0, rank=10,
        parameter_count_b=None, max_context_tokens=None,
    )


class BudgetTests(unittest.TestCase):
    def test_the_default_leaves_headroom_under_the_published_quota(self) -> None:
        # A build is not the only thing spending this: GGUF range reads, model
        # page scrapes and whatever else the session is doing share the window.
        self.assertLess(DEFAULT_BUDGET, HF_REQUESTS_PER_WINDOW)

    def test_a_claim_it_cannot_cover_is_refused_whole(self) -> None:
        # Handing a matcher half its searches would have it report a miss it
        # never established.
        budget = HubRequestBudget(limit=5)

        self.assertTrue(budget.spend(3))
        self.assertFalse(budget.spend(3))
        self.assertEqual(budget.spent, 3)
        self.assertEqual(budget.refused, 3)

    def test_exhaustion_is_reported_once_nothing_is_left(self) -> None:
        budget = HubRequestBudget(limit=2)
        budget.spend(2)

        self.assertTrue(budget.exhausted)

    def test_callers_outside_a_build_are_not_capped(self) -> None:
        budget = UnlimitedBudget()

        self.assertTrue(budget.spend(10_000))
        self.assertFalse(budget.exhausted)


class MatchStoreTests(unittest.TestCase):
    def test_a_hit_and_a_miss_are_both_remembered(self) -> None:
        # The miss is the valuable one: the feed is mostly API-only models and
        # rediscovering that they have no weights is what drained the quota.
        store = RepoMatchStore()
        store.record("glm53", "unsloth/GLM-5.3-GGUF")
        store.record("claudeopus5", None)

        hit = store.get("glm53")
        miss = store.get("claudeopus5")
        assert hit is not None and miss is not None
        self.assertEqual(hit.repo_id, "unsloth/GLM-5.3-GGUF")
        self.assertIsNone(miss.repo_id)

    def test_a_miss_goes_stale_long_before_a_hit(self) -> None:
        # Repository ids do not move, but a model released today may be
        # quantized next week.
        store = RepoMatchStore()
        now = datetime.now(UTC)
        store.record("open", "unsloth/Open-GGUF", now=now)
        store.record("closed", None, now=now)
        later = now + MISS_TTL + timedelta(seconds=1)

        self.assertIsNone(store.get("closed", now=later))
        self.assertIsNotNone(store.get("open", now=later))
        self.assertIsNone(store.get("open", now=now + HIT_TTL + timedelta(seconds=1)))

    def test_it_survives_a_round_trip_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matches.json"
            store = RepoMatchStore()
            store.record("glm53", "unsloth/GLM-5.3-GGUF")
            store.record("claudeopus5", None)
            store.save(path=path)

            reloaded = RepoMatchStore.load(path=path)

            hit = reloaded.get("glm53")
            assert hit is not None
            self.assertEqual(hit.repo_id, "unsloth/GLM-5.3-GGUF")
            self.assertIsNotNone(reloaded.get("claudeopus5"))

    def test_an_unreadable_store_costs_requests_not_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matches.json"
            path.write_text("{ not json", encoding="utf-8")

            self.assertIsNone(RepoMatchStore.load(path=path).get("anything"))


class ResolverIntegrationTests(unittest.TestCase):
    def test_a_remembered_miss_makes_no_request(self) -> None:
        store = RepoMatchStore()
        store.record("someopenmodel", None)

        with patch(
            "llm_launchpad.core.quick_deploy_refresh._find_unsloth_gguf_match"
        ) as matcher:
            resolved = _resolve_aa_model(
                _candidate(), [], hf_api=object(), match_store=store
            )

        self.assertIsNone(resolved)
        matcher.assert_not_called()

    def test_a_fresh_answer_is_written_down(self) -> None:
        store = RepoMatchStore()
        with patch(
            "llm_launchpad.core.quick_deploy_refresh._find_unsloth_gguf_match",
            return_value=None,
        ):
            _resolve_aa_model(_candidate(), [], hf_api=object(), match_store=store)

        remembered = store.get("someopenmodel")
        assert remembered is not None
        self.assertIsNone(remembered.repo_id)

    def test_a_refused_request_is_reported_as_unchecked(self) -> None:
        # Recording a miss here would write down a conclusion no request was
        # made to support, and the next build would trust it.
        store = RepoMatchStore()
        exclusions: list[CatalogExclusion] = []
        with patch(
            "llm_launchpad.core.quick_deploy_refresh._find_unsloth_gguf_match",
            side_effect=HubBudgetExhausted("out"),
        ):
            _resolve_aa_model(
                _candidate(), [], hf_api=object(),
                exclusions=exclusions, match_store=store,
            )

        self.assertIsNone(store.get("someopenmodel"))
        self.assertEqual(len(exclusions), 1)
        self.assertIn("not checked", exclusions[0].reason.lower())


class MatcherSpendTests(unittest.TestCase):
    """The matcher must claim its requests before making them."""

    def test_an_exhausted_budget_stops_the_matcher_before_it_calls_out(self) -> None:
        from llm_launchpad.core.quick_deploy_refresh import _find_unsloth_gguf_match

        budget = HubRequestBudget(limit=0)

        class _Api:
            def model_info(self, **_kwargs: object) -> object:
                raise AssertionError("no request may be made once the budget is spent")

        with self.assertRaises(HubBudgetExhausted):
            _find_unsloth_gguf_match(_candidate(), _Api(), budget)

    def test_the_canonical_probe_is_counted(self) -> None:
        from llm_launchpad.core.quick_deploy_refresh import _find_unsloth_gguf_match

        budget = HubRequestBudget(limit=50)

        class _Api:
            def model_info(self, **_kwargs: object) -> object:
                raise RuntimeError("not found")

        with patch(
            "llm_launchpad.core.quick_deploy_refresh._list_unsloth_gguf_search",
            return_value=[],
        ):
            _find_unsloth_gguf_match(_candidate(), _Api(), budget)

        # A miss is the common case and the one that used to be paid for on
        # every build; it must show up in the accounting.
        self.assertGreater(budget.spent, 0)


class MetadataSpendTests(unittest.TestCase):
    """Reading serving metadata is the other half of what a build spends."""

    def test_metadata_is_charged_to_the_budget(self) -> None:
        from llm_launchpad.core.hf_budget import METADATA_REQUEST_COST
        from llm_launchpad.core.quick_deploy_refresh import _build_resolved_aa_model

        budget = HubRequestBudget(limit=100)
        with patch(
            "llm_launchpad.core.quick_deploy_refresh._fetch_serving_metadata",
            side_effect=RuntimeError("stop here"),
        ):
            _build_resolved_aa_model(_candidate(), [], "unsloth/Some-GGUF", budget=budget)

        self.assertEqual(budget.spent, METADATA_REQUEST_COST)

    def test_an_unaffordable_read_is_unchecked_not_unsized(self) -> None:
        # A throttled read returns an empty weight-size table, which used to
        # surface as "this model's size cannot be determined" -- a statement
        # about the model rather than about the quota that caused it.
        from llm_launchpad.core.quick_deploy_refresh import (
            UNCHECKED_BUDGET_REASON,
            _build_resolved_aa_model,
        )

        budget = HubRequestBudget(limit=0)
        exclusions: list[CatalogExclusion] = []
        with patch(
            "llm_launchpad.core.quick_deploy_refresh._fetch_serving_metadata"
        ) as fetch:
            resolved = _build_resolved_aa_model(
                _candidate(), [], "unsloth/Some-GGUF",
                exclusions=exclusions, budget=budget,
            )

        self.assertIsNone(resolved)
        fetch.assert_not_called()
        self.assertEqual([e.reason for e in exclusions], [UNCHECKED_BUDGET_REASON])


class UnverifiedMissTests(unittest.TestCase):
    """A lookup that never completed must not be written down as an answer."""

    def test_all_probes_failing_is_unknown_not_absent(self) -> None:
        from llm_launchpad.core.hf_budget import HubLookupIncomplete
        from llm_launchpad.core.quick_deploy_refresh import _find_unsloth_gguf_match

        class _Api:
            def model_info(self, **_kwargs: object) -> object:
                raise RuntimeError("429")

        with patch(
            "llm_launchpad.core.quick_deploy_refresh._list_unsloth_gguf_search",
            side_effect=RuntimeError("429"),
        ), self.assertRaises(HubLookupIncomplete):
            _find_unsloth_gguf_match(_candidate(), _Api(), HubRequestBudget(limit=50))

    def test_a_completed_search_that_finds_nothing_is_a_real_miss(self) -> None:
        from llm_launchpad.core.quick_deploy_refresh import _find_unsloth_gguf_match

        class _Api:
            def model_info(self, **_kwargs: object) -> object:
                raise RuntimeError("no such repo")

        # The searches answered; they just had nothing. That is an answer.
        with patch(
            "llm_launchpad.core.quick_deploy_refresh._list_unsloth_gguf_search",
            return_value=[],
        ):
            result = _find_unsloth_gguf_match(
                _candidate(), _Api(), HubRequestBudget(limit=50)
            )

        self.assertIsNone(result)

    def test_an_incomplete_lookup_leaves_the_store_untouched(self) -> None:
        from llm_launchpad.core.hf_budget import HubLookupIncomplete

        store = RepoMatchStore()
        exclusions: list[CatalogExclusion] = []
        with patch(
            "llm_launchpad.core.quick_deploy_refresh._find_unsloth_gguf_match",
            side_effect=HubLookupIncomplete("unknown"),
        ):
            _resolve_aa_model(
                _candidate(), [], hf_api=object(),
                exclusions=exclusions, match_store=store,
            )

        # Believing this for three days is what emptied a whole size category.
        self.assertIsNone(store.get("someopenmodel"))
        self.assertEqual(len(exclusions), 1)
        self.assertIn("not checked", exclusions[0].reason.lower())
