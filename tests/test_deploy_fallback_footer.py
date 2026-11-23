"""Exercise the deployment monitor through phase boundaries and fallback."""

from __future__ import annotations

import unittest

from textual.app import App
from textual.widgets import Static

from llm_launchpad.protocol.enums import BackendType, ComputeProvider, DeploymentState, OperationType
from llm_launchpad.protocol.events import DeploymentAttemptStartedEvent, LogEvent, OperationCompleteEvent, StateChangeEvent
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.workers import _dispatch_event


class CompletionFooterTests(unittest.IsolatedAsyncioTestCase):
    async def test_modal_app_creation_keeps_the_gpu_wait_and_startup_visible(self) -> None:
        app = App[None]()
        async with app.run_test(size=(105, 28)) as pilot:
            screen = MonitorScreen(
                deploy_backend=BackendType.LLAMACPP, summarize_backend_logs=True, show_debug_logs=False
            )
            app.push_screen(screen)
            await pilot.pause()
            _dispatch_event(screen, DeploymentAttemptStartedEvent(gpu_type="L4", gpu_count=5))
            _dispatch_event(screen, LogEvent(line="App deployed in 1.5s!"))
            _dispatch_event(screen, OperationCompleteEvent(), phase_only=True)
            _dispatch_event(screen, StateChangeEvent(
                current=DeploymentState.WARMING_UP,
                operation=OperationType.WARMUP,
                detail="Probing https://example.test",
                timeout_seconds=1800,
            ))
            await pilot.pause()
            title = screen.query_one("#deploy-progress-title", Static)
            detail = screen.query_one("#deploy-progress-detail", Static)
            stages = screen.query_one("#deploy-progress-stages", Static)
            self.assertIn("Waiting for server response", str(title.render()))
            self.assertIn("Warmup timeout in", str(detail.render()))
            self.assertFalse(screen._done)

            _dispatch_event(screen, StateChangeEvent(
                current=DeploymentState.QUEUED,
                operation=OperationType.WARMUP,
                detail="Waiting for GPU scheduling (GPU_L4)",
            ))
            await pilot.pause()
            self.assertIn("Waiting for 5 × L4 GPUs", str(title.render()))
            self.assertIn("✓ Deploy app", str(stages.render()))
            self.assertIn("▸ Get GPUs", str(stages.render()))
            self.assertIn("◦ Start server", str(stages.render()))
            self.assertIn("No ETA", str(detail.render()))

            _dispatch_event(screen, LogEvent(line="load_tensors: loading model tensors"))
            await pilot.pause()
            self.assertIn("Loading model weights", str(title.render()))
            self.assertIn("✓ Get GPUs", str(stages.render()))
            self.assertIn("▸ Start server", str(stages.render()))

            _dispatch_event(screen, OperationCompleteEvent(operation=OperationType.WARMUP), phase_only=True)
            await pilot.pause()
            self.assertIn("▸ Verify", str(stages.render()))
            self.assertFalse(screen._done)
            _dispatch_event(screen, OperationCompleteEvent())
            await pilot.pause()
            self.assertIn("Ready", str(title.render()))
            self.assertIn("✓ Verify", str(stages.render()))

    async def test_failed_attempt_leaves_footer_live_for_the_next_placement(self) -> None:
        app = App[None]()
        async with app.run_test() as pilot:
            screen = MonitorScreen(deploy_backend=BackendType.LLAMACPP, summarize_backend_logs=True)
            app.push_screen(screen)
            await pilot.pause()
            _dispatch_event(screen, DeploymentAttemptStartedEvent(gpu_type="L4", gpu_count=5))
            _dispatch_event(
                screen,
                OperationCompleteEvent(success=False, detail="Placement unavailable"),
                phase_only=True,
            )
            await pilot.pause()
            self.assertFalse(screen._done)
            _dispatch_event(screen, DeploymentAttemptStartedEvent(
                attempt=2, provider=ComputeProvider.VAST, gpu_type="RTX 4090", gpu_count=1
            ))
            await pilot.pause()
            title = screen.query_one("#deploy-progress-title", Static)
            self.assertIn("attempt 2", str(title.render()))
            self.assertEqual(screen._progress.active_stage(), "Checks")
            self.assertEqual(screen._progress.next_stage(), "Get GPUs")
            self.assertNotIn("Deploy app", screen._progress.stages)
