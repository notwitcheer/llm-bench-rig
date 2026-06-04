# Working agreements (agents read this first)

Before architecting or editing this repository, read [`adr/README.md`](adr/README.md)
and the ADRs it indexes. They record this codebase's architectural decisions and
"taste" — follow them.

Decisions with an `enforced_by` path are checked by **pre-commit**: code that
violates them **cannot be committed** (`python3 -m tools.adr.run_all` runs on every
commit). Do not disable or skip the hook.

When you make a new architectural decision, capture it: add an ADR, and — if it is
mechanically checkable — a check + tests, per the "Adding a decision" section of
`adr/README.md`. A decision is not done until it is documented and, where possible,
guarded.

Setup (one-time, on a machine that commits): `python3 -m pip install -r requirements-dev.txt && pre-commit install`.
