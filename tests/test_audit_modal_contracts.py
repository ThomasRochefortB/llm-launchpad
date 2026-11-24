"""Hermetic unit tests for scripts/audit_modal_contracts.py."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_launchpad.core.quick_deploy import QuickDeployProfile
from scripts.audit_modal_contracts import (
    audit_active_catalog,
    audit_modal_docs_text,
    audit_placement_sweep,
    main,
    parse_documented_gpu_limits,
)

_GUIDE_HTML = """
<h2>Specifying GPU type</h2>
<ul>
  <li>T4</li><li>L4</li><li>A10</li><li>A100</li><li>A100-40GB</li>
  <li>A100-80GB</li><li>L40S</li><li>RTX-PRO-6000</li><li>H100</li>
  <li>H200</li><li>B200</li><li>B300</li>
</ul>
<h2>Specifying GPU count</h2>
<p>Currently B300, B200, H200, H100, A100, L4, T4 and L40S instances support up to 8 GPUs,
and A10 instances support up to 4 GPUs.</p>
"""

_PRICING_HTML = """
<h3>GPU Tasks</h3>
<p>Nvidia B300 $0.002500 / sec</p>
<p>Nvidia B200 $0.001736 / sec</p>
<p>Nvidia H200 $0.001261 / sec</p>
<p>Nvidia H100 $0.001097 / sec</p>
<p>Nvidia RTX PRO 6000 $0.000842 / sec</p>
<p>Nvidia A100, 80 GB $0.000694 / sec</p>
<p>Nvidia A100, 40 GB $0.000583 / sec</p>
<p>Nvidia L40S $0.000542 / sec</p>
<p>Nvidia A10 $0.000306 / sec</p>
<p>Nvidia L4 $0.000222 / sec</p>
<p>Nvidia T4 $0.000164 / sec</p>
<h3>CPU</h3>
"""


class AuditModalContractsTests(unittest.TestCase):
    def test_documented_gpu_limits_parser_reads_8_and_4_card_clauses(self) -> None:
        limits = parse_documented_gpu_limits(_GUIDE_HTML)
        self.assertEqual(limits["B200"], 8)
        self.assertEqual(limits["A100"], 8)
        self.assertEqual(limits["L40S"], 8)
        self.assertEqual(limits["A10"], 4)

    def test_audit_modal_docs_passes_for_known_gpus_and_flags_new_gpu_or_lowered_limit(self) -> None:
        self.assertEqual(audit_modal_docs_text(_GUIDE_HTML, _PRICING_HTML), [])

        with_new_gpu = _GUIDE_HTML.replace("<li>B300</li>", "<li>B300</li><li>R9000</li>")
        new_gpu_findings = audit_modal_docs_text(with_new_gpu, _PRICING_HTML)
        self.assertTrue(
            any(f.severity == "error" and "R9000" in f.message for f in new_gpu_findings),
            new_gpu_findings,
        )

        with_lower_limit = _GUIDE_HTML.replace("L4, T4 and L40S instances support up to 8 GPUs", "T4 instances support up to 8 GPUs, and L4 instances support up to 4 GPUs")
        limit_findings = audit_modal_docs_text(with_lower_limit, _PRICING_HTML)
        self.assertTrue(
            any(f.severity == "error" and "L4" in f.message for f in limit_findings),
            limit_findings,
        )

    def test_offline_placement_sweep_and_cli_exit_zero(self) -> None:
        findings, rows = audit_placement_sweep()
        self.assertEqual(findings, [])
        self.assertGreaterEqual(len(rows), 5)
        self.assertEqual(main([]), 0)

    def test_audit_active_catalog_flags_non_power_of_two_modal_shape(self) -> None:
        bad_profile = QuickDeployProfile(
            id="bad-6x-a100",
            display_name="Bad Shape",
            repo_id="org/model-GGUF",
            quant="UD-IQ1_M",
            gpu_type="A100-80GB",
            gpu_count=6,
            profile_label="Economy",
            approx_cost_per_hour_usd=14.99,
            max_context_tokens=1_048_576,
            instance_slug_hint="bad",
            summary="Should fail audit.",
            server_args=(),
        )
        with patch("scripts.audit_modal_contracts.list_quick_deploy_profiles", return_value=(bad_profile,)):
            findings, _rows = audit_active_catalog()

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, "error")
        self.assertIn("6x A100-80GB", findings[0].message)


if __name__ == "__main__":
    unittest.main()
