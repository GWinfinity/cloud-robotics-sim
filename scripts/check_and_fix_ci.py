#!/usr/bin/env python3
"""Run local quality checks and apply safe auto-fixes.

This script runs ruff, black, and mypy locally. It will auto-fix lint issues
and format code in the working tree only — this script itself never commits
or pushes anything. (The GitHub Actions auto-fix workflows are what commit
style fixes, via a ``style/auto-fix-*`` branch and a pull request.) Review
the diff before committing it yourself.
"""

import subprocess
import sys


def run_command(cmd: str) -> tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode, result.stdout, result.stderr


def main() -> int:
    """Run quality checks and safe auto-fixes."""
    print("=" * 60)
    print("🔧 Local Quality Check Tool")
    print("=" * 60)

    # Auto-fix lint issues
    print("\n🔧 Running ruff check --fix ...")
    rc, stdout, stderr = run_command(f"{sys.executable} -m ruff check --fix src/ tests/")
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    if rc != 0:
        print("❌ Ruff found unfixable issues.")
        return rc
    print("✅ Ruff passed.")

    # Format code
    print("\n🔧 Running black ...")
    rc, stdout, stderr = run_command(f"{sys.executable} -m black src/ tests/")
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    if rc != 0:
        print("❌ Black formatting failed.")
        return rc
    print("✅ Black passed.")

    # Type check
    print("\n🔧 Running mypy ...")
    rc, stdout, stderr = run_command(f"{sys.executable} -m mypy src/cloud_robotics_sim")
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    if rc != 0:
        print("❌ Mypy found type errors.")
        return rc
    print("✅ Mypy passed.")

    # Tests
    print("\n🔧 Running pytest ...")
    rc, stdout, stderr = run_command(f"{sys.executable} -m pytest tests/ -q")
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    if rc != 0:
        print("❌ Tests failed.")
        return rc
    print("✅ Tests passed.")

    print("\n" + "=" * 60)
    print("🚀 All checks passed. Review the diff before committing.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
