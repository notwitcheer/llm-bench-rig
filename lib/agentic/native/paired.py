"""Paired statistics for a same-task, two-condition A/B (e.g. reasoning on vs off).
Mirrors the closed-form, dependency-free style of `lib.agentic.phaseb.wilson_ci`: no
scipy (not a repo dependency — grepped, unused elsewhere), exact math via `math.comb`."""
import math

from lib.agentic.native.agent_loop import run_agent


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """McNemar's exact test on paired pass/fail outcomes. `a[i]`/`b[i]` are the same
    item's outcome under condition A / condition B. Only the discordant pairs (where
    the two conditions disagree) carry information; concordant pairs are ignored.
    `b01` = a False, b True; `b10` = a True, b False. `p` is the exact two-sided
    binomial test of an (b01, b10) split this extreme under the null that a
    discordant pair is equally likely to go either way (p=0.5)."""
    if len(a) != len(b):
        raise ValueError(f"a and b must be equal length, got {len(a)} and {len(b)}")
    b01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    b10 = sum(1 for x, y in zip(a, b) if x and (not y))
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        p = min(1.0, 2 * sum(math.comb(n, i) * 0.5 ** n for i in range(k + 1)))
    return {"b01": b01, "b10": b10, "n_discordant": n, "p": p}


def run_paired(task: dict, client_a, client_b, tools: list, max_steps: int = 12,
               dispatch=None) -> dict:
    """Run the same task's goal through `run_agent` once per client, so the two
    RunResults are directly comparable (same goal/tools, different condition).
    `client_a` / `client_b` are e.g. reasoning-on / reasoning-off instances of the
    same client class — each drives its own independent trajectory."""
    goal = task["goal"]
    a = run_agent(client_a, goal, tools, max_steps=max_steps, dispatch=dispatch)
    b = run_agent(client_b, goal, tools, max_steps=max_steps, dispatch=dispatch)
    return {"id": task["id"], "a": a, "b": b}
