"""Package smoke test executed against built distributions."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    import llm_launchpad

    assert llm_launchpad.__file__, "Package import did not resolve"

    result = subprocess.run(
        ["llm-launchpad", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode

    if "llm-launchpad CLI" not in result.stdout:
        sys.stderr.write(result.stdout)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
