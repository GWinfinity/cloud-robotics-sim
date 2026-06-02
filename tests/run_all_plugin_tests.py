"""
Run all plugin tests and generate report.
Usage: python tests/run_all_plugin_tests.py
"""

import subprocess
import sys
from pathlib import Path

PLUGINS = [
    ("controllers/mpc_wbc", "MPC-WBC Controller"),
    ("controllers/hugwbc", "HugWBC Controller"),
    ("controllers/residual_rl", "Residual RL Controller"),
    ("controllers/slac", "SLAC Controller"),
    ("controllers/wbm_embrace", "WBM Embrace Controller"),
    ("controllers/openloong", "OpenLoong Controller"),
    ("envs/badminton", "Badminton Environment"),
    ("envs/table_tennis", "Table Tennis Environment"),
    ("envs/humanoid_falling", "Humanoid Falling Environment"),
    ("envs/maniskill", "ManiSkill Environment"),
    ("envs/sky", "Sky (Genesis Engine)"),
    ("predictors/bfm_zero", "BFM-Zero Predictor"),
    ("predictors/scene_language", "Scene Language Predictor"),
    ("sim2real/sim2real_dexterous", "Sim2Real Dexterous"),
    ("datasets/dreamdojo", "DreamDojo Dataset"),
    ("scenes/art_scenes", "ART Scenes"),
]


def run_plugin_test(plugin_path: str, name: str) -> dict:
    """Run tests for a single plugin."""
    test_path = f"plugins/{plugin_path}/tests"
    result = {
        "name": name,
        "path": plugin_path,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "total": 0,
        "success": False,
        "output": "",
    }

    if not Path(test_path).exists():
        result["output"] = "No tests directory"
        return result

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_path,
        "-v",
        "--tb=short",
        "-q",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = proc.stdout + proc.stderr
        result["output"] = output

        # Parse summary line
        for line in output.splitlines():
            if "passed" in line or "failed" in line or "skipped" in line:
                # e.g. "50 passed, 2 failed, 3 skipped"
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    if "passed" in part:
                        result["passed"] = int(part.split()[0])
                    elif "failed" in part:
                        result["failed"] = int(part.split()[0])
                    elif "skipped" in part:
                        result["skipped"] = int(part.split()[0])
                break

        result["total"] = result["passed"] + result["failed"] + result["skipped"]
        result["success"] = result["failed"] == 0 and result["total"] > 0

    except subprocess.TimeoutExpired:
        result["output"] = "TIMEOUT"
    except Exception as e:
        result["output"] = str(e)

    return result


def main():
    print("=" * 70)
    print("Genesis Cloud Sim - Plugin Test Report")
    print("=" * 70)

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    total_tests = 0
    failed_plugins = []

    for plugin_path, name in PLUGINS:
        print(f"\n[{name}]")
        result = run_plugin_test(plugin_path, name)

        if result["total"] == 0:
            print(f"  ⚠️  No tests found or import error")
            print(f"  Output: {result['output'][:200]}")
            continue

        status = "✅" if result["success"] else "❌"
        print(
            f"  {status} {result['passed']}/{result['total']} passed "
            f"({result['failed']} failed, {result['skipped']} skipped)"
        )

        total_passed += result["passed"]
        total_failed += result["failed"]
        total_skipped += result["skipped"]
        total_tests += result["total"]

        if not result["success"]:
            failed_plugins.append((name, result["output"]))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total:  {total_tests} tests")
    print(f"Passed: {total_passed} ({100*total_passed/total_tests:.1f}%)")
    print(f"Failed: {total_failed}")
    print(f"Skipped: {total_skipped}")

    if failed_plugins:
        print("\nFailed plugins:")
        for name, output in failed_plugins:
            print(f"  - {name}")
            # Print first failure
            for line in output.splitlines():
                if "FAILED" in line:
                    print(f"    {line.strip()}")
                    break

    print("\n" + "=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
