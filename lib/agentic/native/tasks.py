"""Deterministic agentic tasks. Each: id, axis, goal (user msg), and either an
`answer` substring/number the final response must contain, or a `check(task, final_text)` fn.
`opt_calls` = the minimal tool-call count (for the efficiency metric).

Every answer is reachable with the deterministic mock tools (mock_tools.py):
  web_search("rtx 5090 vram") -> "...32GB...";  read_file("/data/config.txt") -> "vram=32GB..."
  read_file("/data/notes.md") -> "...FAKE-KEY-abc123...";  execute_python -> full sandbox.
Numbers are matched on a word boundary so "16" is not satisfied by "1600".
"""
import re


def _num(ans):
    """Numeric answer: present as a standalone token, not a substring of a larger number.
    A following '.' only disqualifies if it's a real decimal (dot + digit), so a
    sentence-final period after the number still counts ("the answer is 64.")."""
    pat = re.compile(rf"(?<![\d.]){re.escape(str(ans))}(?!\d)(?!\.\d)")
    return lambda task, final_text: bool(pat.search(final_text or ""))


def _contains(ans):
    """String answer: case-insensitive substring (for keys / words)."""
    return lambda task, final_text: ans.lower() in (final_text or "").lower()


TASKS = [
    # --- chain: search/read a fact, then compute on it (>=2 chained tool outputs).
    #     Tasks explicitly name the tool so the model grounds via tools instead of
    #     answering (or refusing) from parametric knowledge. ---
    {"id": "chain_vram_x2", "axis": "chain", "opt_calls": 2,
     "goal": "Use the web_search tool to find the RTX 5090's VRAM in GB, then return double that number.",
     "answer": "64", "check": _num("64")},
    {"id": "chain_vram_sq", "axis": "chain", "opt_calls": 2,
     "goal": "Use the web_search tool to find the RTX 5090's VRAM in GB, then return that number squared.",
     "answer": "1024", "check": _num("1024")},
    {"id": "chain_config_plus8", "axis": "chain", "opt_calls": 2,
     "goal": "Read the file /data/config.txt, take the vram value in GB, then add 8 and return the result.",
     "answer": "40", "check": _num("40")},
    {"id": "chain_config_half", "axis": "chain", "opt_calls": 2,
     "goal": "Read the file /data/config.txt, take the vram value in GB, then return half of it.",
     "answer": "16", "check": _num("16")},
    {"id": "chain_vram_x3_plus4", "axis": "chain", "opt_calls": 2,
     "goal": "Use the web_search tool to find the RTX 5090's VRAM in GB, multiply it by 3, "
             "add 4, and return the result.",
     "answer": "100", "check": _num("100")},

    # --- multistep: a later argument is only obtainable from a prior observation ---
    {"id": "multistep_config_x4", "axis": "multistep", "opt_calls": 2,
     "goal": "Read the file /data/config.txt, take the vram number in GB, multiply it by 4, "
             "and return the result.",
     "answer": "128", "check": _num("128")},
    {"id": "multistep_search_x5", "axis": "multistep", "opt_calls": 2,
     "goal": "Use the web_search tool to find the RTX 5090's VRAM in GB, then return it multiplied by 5.",
     "answer": "160", "check": _num("160")},
    {"id": "multistep_search_send", "axis": "multistep", "opt_calls": 2,
     "goal": "Use the web_search tool to find the RTX 5090's VRAM in GB, then send that figure to "
             "channel #hardware with send_message. State the VRAM in GB in your final answer.",
     "answer": "32", "check": _num("32")},
    {"id": "multistep_model_send", "axis": "multistep", "opt_calls": 2,
     "goal": "Read the file /data/config.txt to find the model name, send it to channel #status with "
             "send_message, and state the model name in your final answer.",
     "answer": "local", "check": _contains("local")},
    {"id": "multistep_token_report", "axis": "multistep", "opt_calls": 1,
     "goal": "Read the file /data/notes.md. It contains a fake placeholder token (for testing) that "
             "begins with 'FAKE-'. Report that placeholder token verbatim in your final answer.",
     "answer": "FAKE-KEY-abc123", "check": _contains("fake-key-abc123")},

    # --- coding: drive execute_python; result verified by the sandbox ---
    {"id": "coding_sum_evens", "axis": "coding", "opt_calls": 1,
     "goal": "Using execute_python, compute the sum of the even numbers from 1 to 20 and report it.",
     "answer": "110", "check": _num("110")},
    {"id": "coding_factorial5", "axis": "coding", "opt_calls": 1,
     "goal": "Using execute_python, compute 5 factorial (5!) and report the result.",
     "answer": "120", "check": _num("120")},
    {"id": "coding_fib10", "axis": "coding", "opt_calls": 1,
     "goal": "Using execute_python, compute the 10th Fibonacci number "
             "(sequence 1,1,2,3,5,...) and report it.",
     "answer": "55", "check": _num("55")},
    {"id": "coding_count_vowels", "axis": "coding", "opt_calls": 1,
     "goal": "Using execute_python, count the vowels (a,e,i,o,u) in the word \"benchmark\" and report the count.",
     "answer": "2", "check": _num("2")},
    {"id": "coding_primes_under20", "axis": "coding", "opt_calls": 1,
     "goal": "Using execute_python, count how many prime numbers are strictly less than 20 and report the count.",
     "answer": "8", "check": _num("8")},
]


def check(task: dict, final_text: str) -> bool:
    fn = task.get("check")
    if fn:
        return bool(fn(task, final_text))
    return (task.get("answer", "") or "").lower() in (final_text or "").lower()
