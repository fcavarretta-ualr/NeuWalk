#!/usr/bin/env python3
"""Run ``python3 test.py N`` for every integer N from 0 through 200."""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_test(number: int) -> tuple[int, int, str, str]:
    """Run test.py for one number and return its result."""
    result = subprocess.run(
        ["python3", "test.py", str(number)],
        capture_output=True,
        text=True,
        check=False,
    )
    return number, result.returncode, result.stdout, result.stderr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run test.py with every integer from 0 through 200."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=14,
        help="maximum number of simultaneous processes (default: 10)",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_test, n): n for n in range(200)}

        for future in as_completed(futures):
            number = futures[future]
            try:
                _, returncode, stdout, stderr = future.result()
            except Exception as exc:
                failures += 1
                print(f"[{number}] ERROR: {exc}")
                continue

            status = "OK" if returncode == 0 else f"FAILED ({returncode})"
            print(f"[{number}] {status}")
            if stdout:
                print(stdout, end="" if stdout.endswith("\n") else "\n")
            if stderr:
                print(stderr, end="" if stderr.endswith("\n") else "\n")
            failures += returncode != 0

    print(f"Finished: {201 - failures} succeeded, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
