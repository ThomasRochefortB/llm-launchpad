from __future__ import annotations

import unittest
from importlib.metadata import PackageNotFoundError, version

import llm_launchpad

from llm_launchpad._version import __version__ as source_version


class ReleaseMetadataTests(unittest.TestCase):
    def test_package_exports_single_source_version(self) -> None:
        self.assertEqual(llm_launchpad.__version__, source_version)

    def test_distribution_metadata_matches_package_version(self) -> None:
        try:
            dist_version = version("llm-launchpad")
        except PackageNotFoundError:
            self.skipTest("Package metadata is unavailable in this environment.")

        self.assertEqual(dist_version, llm_launchpad.__version__)
