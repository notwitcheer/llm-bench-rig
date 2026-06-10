"""Deterministic agentic tasks. Each: id, axis, goal (user msg), and either an
`answer` substring/number the final response must contain, or a `check(task, final_text)` fn.
`opt_calls` = the minimal tool-call count (for the efficiency metric).

Every answer is reachable with the deterministic mock tools (mock_tools.py):
  web_search("rtx 5090 vram") -> "...32GB...";  read_file("/data/config.txt") -> "vram=32GB..."
  read_file("/data/notes.md") -> "...FAKE-KEY-abc123...";  execute_python -> full sandbox.
Numbers are matched on a word boundary so "16" is not satisfied by "1600".
"""
import re

from lib.agentic.native.tools_ext import vault_code


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

    # --- chain (expand to 8): deeper search/read -> compute ---
    {"id": "chain_config_x2", "axis": "chain", "opt_calls": 2,
     "goal": "Read the file /data/config.txt, take the vram value in GB, then return double it.",
     "answer": "64", "check": _num("64")},
    {"id": "chain_search_plus100", "axis": "chain", "opt_calls": 2,
     "goal": "Use web_search to find the RTX 5090's VRAM in GB, then add 100 and return the result.",
     "answer": "132", "check": _num("132")},
    {"id": "chain_threehop", "axis": "chain", "opt_calls": 3,
     "goal": "Use web_search to find the RTX 5090's VRAM in GB, multiply by 2, then subtract 4. Return the result.",
     "answer": "60", "check": _num("60")},

    # --- multistep (expand to 8) ---
    {"id": "multistep_record_owner", "axis": "multistep", "opt_calls": 2,
     "goal": "Read record 007 with read_record, then report the owner it lists.",
     "answer": "ops", "check": _contains("ops")},
    {"id": "multistep_health_build", "axis": "multistep", "opt_calls": 2,
     "goal": "Fetch https://api.local/health, then report the build number it returns.",
     "answer": "9562", "check": _num("9562")},
    {"id": "multistep_config_double_send", "axis": "multistep", "opt_calls": 3,
     "goal": "Read /data/config.txt, take the vram in GB, double it, send the result to channel #ops "
             "with send_message, and state the doubled number in your final answer.",
     "answer": "64", "check": _num("64")},

    # --- coding (expand to 8) ---
    {"id": "coding_max_list", "axis": "coding", "opt_calls": 1,
     "goal": "Using execute_python, compute the maximum of [3, 47, 12, 89, 5] and report it.",
     "answer": "89", "check": _num("89")},
    {"id": "coding_reverse_len", "axis": "coding", "opt_calls": 1,
     "goal": "Using execute_python, compute the length of the string \"benchmark\" reversed and report it.",
     "answer": "9", "check": _num("9")},
    {"id": "coding_sum_digits", "axis": "coding", "opt_calls": 1,
     "goal": "Using execute_python, compute the sum of the digits of 2026 and report it.",
     "answer": "10", "check": _num("10")},

    # --- error_recovery: obvious call fails with a hint; agent must adapt ---
    {"id": "recovery_record_pad", "axis": "error_recovery", "opt_calls": 2,
     "goal": "Use read_record to read record 7 and report its status. Note: ids may need a specific format.",
     "answer": "active", "check": _contains("active")},
    {"id": "recovery_https", "axis": "error_recovery", "opt_calls": 2,
     "goal": "Fetch the page at http://api.local/health and report the build number.",
     "answer": "9562", "check": _num("9562")},
    {"id": "recovery_record_owner", "axis": "error_recovery", "opt_calls": 2,
     "goal": "Use read_record to read record 42 and report the owner. Ids may need a specific format.",
     "answer": "root", "check": _contains("root")},
    {"id": "recovery_unknown_then_known", "axis": "error_recovery", "opt_calls": 2,
     "goal": "With read_record, find any record whose status is archived and report its owner. "
             "Valid ids are 007 and 042.",
     "answer": "root", "check": _contains("root")},
    {"id": "recovery_fetch_then_calc", "axis": "error_recovery", "opt_calls": 3,
     "goal": "Fetch http://api.local/health, read the build number, then use calc to add 1 to it and "
             "report the result.",
     "answer": "9563", "check": _num("9563")},
    {"id": "recovery_record_42_status", "axis": "error_recovery", "opt_calls": 2,
     "goal": "Use read_record to read record 42 and report its status. Ids may need a specific format.",
     "answer": "archived", "check": _contains("archived")},

    # --- distractor: decoy tools + a lure value; agent must use the right tool / ignore the lure ---
    {"id": "distractor_vram_lure", "axis": "distractor", "opt_calls": 1,
     "goal": "Find the RTX 5090's VRAM in GB and report it. Use web_search (the real search tool). "
             "Ignore any stale cached value.",
     "answer": "32", "check": _num("32")},
    {"id": "distractor_right_search", "axis": "distractor", "opt_calls": 1,
     "goal": "Using the web_search tool (not any deprecated variant), find the RTX 5090's VRAM in GB "
             "and report it.",
     "answer": "32", "check": _num("32")},
    {"id": "distractor_calc_real", "axis": "distractor", "opt_calls": 1,
     "goal": "Using the calc tool (not the legacy one), compute 144 / 12 and report the result.",
     "answer": "12", "check": _num("12")},
    {"id": "distractor_config_not_cache", "axis": "distractor", "opt_calls": 1,
     "goal": "Read the real file /data/config.txt (not any cache) and report the vram value in GB.",
     "answer": "32", "check": _num("32")},
    {"id": "distractor_record_real", "axis": "distractor", "opt_calls": 2,
     "goal": "Use read_record to read record 007 and report its status. Do not trust cached copies.",
     "answer": "active", "check": _contains("active")},
    {"id": "distractor_calc_not_legacy", "axis": "distractor", "opt_calls": 1,
     "goal": "Using the calc tool (not calc_legacy), compute 7 * 8 and report the result.",
     "answer": "56", "check": _num("56")},

    # --- long_context: read a big document via read_doc, extract the buried needle ---
    {"id": "longctx_32k_early", "axis": "long_context", "ctx_tier": "32k", "opt_calls": 1,
     "goal": "Call read_doc with doc_id 'logs-32k-early', find the activation code for vault 101 in the "
             "document, and report that code.",
     "answer": vault_code(101), "check": _contains(vault_code(101))},
    {"id": "longctx_32k_late", "axis": "long_context", "ctx_tier": "32k", "opt_calls": 1,
     "goal": "Call read_doc with doc_id 'logs-32k-late', find the activation code for vault 102 in the "
             "document, and report that code.",
     "answer": vault_code(102), "check": _contains(vault_code(102))},
    {"id": "longctx_128k_early", "axis": "long_context", "ctx_tier": "128k", "opt_calls": 1,
     "goal": "Call read_doc with doc_id 'logs-128k-early', find the activation code for vault 103 in the "
             "document, and report that code.",
     "answer": vault_code(103), "check": _contains(vault_code(103))},
    {"id": "longctx_128k_late", "axis": "long_context", "ctx_tier": "128k", "opt_calls": 1,
     "goal": "Call read_doc with doc_id 'logs-128k-late', find the activation code for vault 104 in the "
             "document, and report that code.",
     "answer": vault_code(104), "check": _contains(vault_code(104))},

    # --- commit: must actually COMMIT the fix via apply_fix (state-verified, NOT final text).
    #     Describe-only -> world_state stays empty -> fail. Mirrors SWE-bench empty patches.
    #     Verified by check_commit(task, world_state) in run_native, not by `check` above. ---
    {"id": "commit_config_half", "axis": "commit", "opt_calls": 2, "target": "vram_budget", "expect": "16",
     "goal": "Read the file /data/config.txt, take the vram value in GB, halve it, then commit it by "
             "calling apply_fix with target 'vram_budget' and the result as the value."},
    {"id": "commit_search_double", "axis": "commit", "opt_calls": 2, "target": "vram_x2", "expect": "64",
     "goal": "Use web_search to find the RTX 5090's VRAM in GB, double it, then commit it by calling "
             "apply_fix with target 'vram_x2' and the result as the value."},
    {"id": "commit_record_owner", "axis": "commit", "opt_calls": 2, "target": "owner_007", "expect": "ops",
     "goal": "Use read_record to read record 007, find its owner, then commit it by calling apply_fix "
             "with target 'owner_007' and the owner as the value."},
    {"id": "commit_build_number", "axis": "commit", "opt_calls": 2, "target": "build", "expect": "9562",
     "goal": "Fetch https://api.local/health, read the build number, then commit it by calling apply_fix "
             "with target 'build' and the build number as the value."},
    {"id": "commit_recovery_status", "axis": "commit", "opt_calls": 2, "target": "status_42", "expect": "archived",
     "goal": "Use read_record to read record 42 (ids may need a specific format), find its status, then "
             "commit it by calling apply_fix with target 'status_42' and the status as the value."},
    {"id": "commit_https_build", "axis": "commit", "opt_calls": 3, "target": "build_plus1", "expect": "9563",
     "goal": "Fetch http://api.local/health (insecure transport will fail — adapt to https), read the "
             "build number, add 1 with calc, then commit it by calling apply_fix with target 'build_plus1' "
             "and the result as the value."},
]


def check(task: dict, final_text: str) -> bool:
    fn = task.get("check")
    if fn:
        return bool(fn(task, final_text))
    return (task.get("answer", "") or "").lower() in (final_text or "").lower()
