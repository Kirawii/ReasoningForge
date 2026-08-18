#!/usr/bin/env python3
"""AutoDL/local deployment adaptation: validate the 2x RTX 5090 runtime."""

from __future__ import annotations

import argparse

from cs336_alignment.local_utils import (
    collect_local_environment,
    print_local_environment,
    validate_local_environment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when a required version or capability is missing.",
    )
    parser.add_argument(
        "--require-rtx-5090",
        action="store_true",
        help="Also require both visible GPU names to contain 'RTX 5090'.",
    )
    args = parser.parse_args()

    environment = collect_local_environment()
    print_local_environment(environment)
    issues = validate_local_environment(
        environment,
        require_rtx_5090=args.require_rtx_5090,
    )
    if issues:
        print("Environment issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1 if args.strict else 0
    print("Environment status: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
