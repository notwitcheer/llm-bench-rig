import pytest

from lib.agentic.native.paired import mcnemar, run_paired
from lib.agentic.native.agent_loop import RunResult


class ScriptedClient:
    """Returns pre-baked assistant messages; records the tool results it was fed.
    Copied from tests/test_native_agent_loop.py so each client instance in a pair
    is independently scripted."""
    def __init__(self, turns):
        self.turns = list(turns)
        self.seen_tool_msgs = []

    def chat(self, messages, tools):
        for m in messages:
            if m.get("role") == "tool":
                self.seen_tool_msgs.append(m["content"])
        return self.turns.pop(0)


# ---------------------------------------------------------------------------
# mcnemar
# ---------------------------------------------------------------------------

def test_mcnemar_hand_computed_case():
    # b10 = 8 pairs where a=True,b=False; b01 = 2 pairs where a=False,b=True.
    # n = 10 discordant pairs.
    # Hand calc: p = 2 * sum(C(10,i) for i in 0..2) * 0.5**10
    #              = 2 * (C(10,0) + C(10,1) + C(10,2)) / 1024
    #              = 2 * (1 + 10 + 45) / 1024
    #              = 2 * 56 / 1024
    #              = 112 / 1024
    #              = 0.109375
    a = [True] * 8 + [False] * 2
    b = [False] * 8 + [True] * 2
    result = mcnemar(a, b)
    assert result["b01"] == 2
    assert result["b10"] == 8
    assert result["n_discordant"] == 10
    assert result["p"] == pytest.approx(0.109375, abs=1e-9)
    # not significant at alpha=0.05 for this particular split
    assert result["p"] >= 0.05


def test_mcnemar_all_concordant_gives_p_one():
    a = [True, True, False, False]
    b = [True, True, False, False]
    result = mcnemar(a, b)
    assert result["b01"] == 0
    assert result["b10"] == 0
    assert result["n_discordant"] == 0
    assert result["p"] == 1.0


def test_mcnemar_symmetric_split_gives_p_one():
    # b01 == b10 == 3 (n=6): p = 2 * sum(C(6,i) for i in 0..3) / 64
    #                          = 2 * (1+6+15+20) / 64 = 84/64 = 1.3125 -> clipped to 1.0
    a = [True] * 3 + [False] * 3
    b = [False] * 3 + [True] * 3
    result = mcnemar(a, b)
    assert result["b01"] == 3
    assert result["b10"] == 3
    assert result["p"] == 1.0


def test_mcnemar_unequal_length_raises():
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


# ---------------------------------------------------------------------------
# run_paired
# ---------------------------------------------------------------------------

def test_run_paired_drives_each_client_independently():
    task = {"id": "t-paired-1", "goal": "double 21"}

    turns_a = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "calc", "arguments": '{"expr": "21*2"}'}}]},
        {"role": "assistant", "content": "answer from A: 42", "tool_calls": None},
    ]
    turns_b = [
        {"role": "assistant", "content": "answer from B: 42 (no tools)", "tool_calls": None},
    ]

    def dispatch(name, args):
        return {"ok": True, "result": "tool-result", "error": ""}

    client_a = ScriptedClient(turns_a)
    client_b = ScriptedClient(turns_b)

    result = run_paired(task, client_a, client_b, tools=[], max_steps=5, dispatch=dispatch)

    assert result["id"] == "t-paired-1"
    assert isinstance(result["a"], RunResult)
    assert isinstance(result["b"], RunResult)
    assert result["a"].final_text == "answer from A: 42"
    assert result["b"].final_text == "answer from B: 42 (no tools)"
    # a used a tool, b did not -> proof the two runs are independent, not shared state
    assert result["a"].n_tool_calls == 1
    assert result["b"].n_tool_calls == 0
    assert client_a.turns == []
    assert client_b.turns == []
