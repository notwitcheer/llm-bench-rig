# Architectural Decision Records

Durable record of this codebase's architectural decisions. **Agents: read these
before architecting or editing the repo.** Decisions with an `enforced_by` path
are checked by pre-commit — violating code cannot be committed.

| ID | Decision | Status | Enforced by |
|----|----------|--------|-------------|
| [ADR-0001](ADR-0001-no-stop-sequences-code-eval.md) | No stop sequences in the code-eval harness | accepted | `tools/adr/check_no_stop_sequences.py` |
| [ADR-0002](ADR-0002-results-record-think-mode.md) | Every benchmark result records think mode | accepted | `tools/adr/check_think_recorded.py` |
| [ADR-0003](ADR-0003-invariants-in-tested-code.md) | Invariants live in tested deterministic helpers | accepted | convention |

## Adding a decision

When a new architectural decision is made ("enforce X, add it to the ADR"):

1. Write `adr/ADR-NNNN-<slug>.md` (copy the frontmatter from an existing one).
2. If the decision is mechanically checkable, add `tools/adr/check_<name>.py`
   exposing `find_violations(source, filename) -> list[str]` and `main() -> int`,
   then register it in `tools/adr/run_all.py`'s `CHECKS` list.
3. Add its tests to `tests/test_adr_checks.py`: one asserting the real file is
   clean, one feeding a synthetic violation and asserting it is caught.
4. Add a row to the table above.

A decision is not "done" until it is documented and (where possible) guarded.

## Running the checks

`python3 -m tools.adr.run_all` (also runs automatically on `git commit` via
pre-commit). Setup: `python3 -m pip install -r requirements-dev.txt && pre-commit install`.
