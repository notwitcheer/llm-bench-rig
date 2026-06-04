---
id: ADR-0003
title: Invariants live in tested deterministic helpers, not LLM prompts
status: accepted
date: 2026-06-04
enforced_by: convention
---
## Context
State transitions, format validation, and correctness invariants are unreliable
when left to an LLM. The publishing cadence's `queue_ops.py` (peek/pop/add +
format validation) and the grounding/slop gate both proved this: the model picks
the topic and voice, tested code guarantees the queue never gets a malformed line
and a draft is popped exactly once.

## Decision
The LLM is for judgment (what to write, which topic, which voice). Every invariant
— state transitions, validation, anything where "wrong" is a real cost — lives in
a deterministic helper with unit tests that the model calls. Do not encode an
invariant as a prompt instruction and hope.

## Consequences
Invariants survive prompt drift and model swaps. Behavior is reproducible and testable.

## Enforcement
Convention + existing unit tests (e.g. `cadence/` `queue_ops` tests). No dedicated
lint rule — this is a design principle, not a single-file pattern. New invariants
should ship with their own tests and, where statically checkable, an ADR + check.
