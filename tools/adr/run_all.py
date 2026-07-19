"""Run every ADR check. Exit 1 if any enforced decision is violated.

Invoked by pre-commit (`python -m tools.adr.run_all`) and importable by tests.
"""
import sys

from . import check_mc_gate_wired, check_no_stop_sequences, check_think_recorded

CHECKS = [check_no_stop_sequences, check_think_recorded, check_mc_gate_wired]


def main() -> int:
    failed = 0
    for check in CHECKS:
        if check.main() != 0:
            failed = 1
    if failed:
        print(
            "\nADR enforcement failed — commit blocked. Fix the violations above, "
            "or update the relevant ADR if the decision genuinely changed.",
            file=sys.stderr,
        )
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
