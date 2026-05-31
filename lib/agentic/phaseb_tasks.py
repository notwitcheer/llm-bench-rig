"""Phase B task battery (v2) — harder, multi-step, discriminating agent tasks.

Each task: setup (shell, builds fixtures under /tmp/phaseb), prompt (given to
`hermes -z`), check (shell, returns 0 iff the resulting filesystem state is
correct). Tasks are multi-step and carry "traps" (filtering, headers, noise,
sorting, dedup, nesting, chaining) so capable models actually differ — measured
by completion AND by turns/tokens (parsed from the llama-server log).

Raw strings keep `\\n` literal for printf. Checks strip noise to tolerate prose.
"""

def _eq(path, val):
    return f"""test "$(tr -dc '0-9a-zA-Z.@_-' < {path} 2>/dev/null)" = '{val}'"""

def _num(path, val):
    return f"""test "$(tr -dc '0-9-' < {path} 2>/dev/null)" = '{val}'"""

TASKS = [
    # --- baseline sanity ---
    {"id": "sum_csv",
     "setup": r"mkdir -p /tmp/phaseb && printf '12\n30\n8\n50\n' > /tmp/phaseb/input.csv && rm -f /tmp/phaseb/out.txt",
     "prompt": "Read /tmp/phaseb/input.csv (one integer per line). Sum the integers and write ONLY the total number to /tmp/phaseb/out.txt.",
     "check": _num("/tmp/phaseb/out.txt", "100")},

    # --- filtering traps ---
    {"id": "sum_even",
     "setup": r"mkdir -p /tmp/phaseb && printf '3\n8\n5\n10\n7\n12\n' > /tmp/phaseb/nums.txt && rm -f /tmp/phaseb/even.txt",
     "prompt": "Read /tmp/phaseb/nums.txt (one integer per line). Sum ONLY the even numbers and write ONLY that total to /tmp/phaseb/even.txt.",
     "check": _num("/tmp/phaseb/even.txt", "30")},  # 8+10+12

    {"id": "sum_with_noise",
     "setup": r"mkdir -p /tmp/phaseb && printf '10\nhello\n20\nworld\n5\n#comment\n15\n' > /tmp/phaseb/mixed.txt && rm -f /tmp/phaseb/clean.txt",
     "prompt": "Read /tmp/phaseb/mixed.txt. Some lines are integers and some are text. Sum ONLY the lines that are integers and write ONLY that total to /tmp/phaseb/clean.txt.",
     "check": _num("/tmp/phaseb/clean.txt", "50")},  # 10+20+5+15

    {"id": "sum_column_header",
     "setup": r"mkdir -p /tmp/phaseb && printf 'name,score\na,10\nb,25\nc,15\n' > /tmp/phaseb/scores.csv && rm -f /tmp/phaseb/colsum.txt",
     "prompt": "Read the CSV /tmp/phaseb/scores.csv. It has a header row (name,score). Sum the values in the 'score' column (do NOT include the header) and write ONLY that total to /tmp/phaseb/colsum.txt.",
     "check": _num("/tmp/phaseb/colsum.txt", "50")},  # 10+25+15

    # --- sorting / logic ---
    {"id": "second_largest",
     "setup": r"mkdir -p /tmp/phaseb && printf '4\n42\n17\n42\n9\n38\n' > /tmp/phaseb/vals.txt && rm -f /tmp/phaseb/second.txt",
     "prompt": "Read /tmp/phaseb/vals.txt (one integer per line). Find the SECOND LARGEST DISTINCT value and write ONLY that number to /tmp/phaseb/second.txt.",
     "check": _num("/tmp/phaseb/second.txt", "38")},  # distinct: 42,38,17,9,4 -> 2nd = 38

    {"id": "unique_words",
     "setup": r"mkdir -p /tmp/phaseb && printf 'red blue red green\nblue red yellow\n' > /tmp/phaseb/words.txt && rm -f /tmp/phaseb/uniq.txt",
     "prompt": "Read /tmp/phaseb/words.txt. Count the number of DISTINCT (unique) whitespace-separated words and write ONLY that count to /tmp/phaseb/uniq.txt.",
     "check": _num("/tmp/phaseb/uniq.txt", "4")},  # red blue green yellow

    {"id": "count_substring",
     "setup": r"mkdir -p /tmp/phaseb && printf 'INFO ok\nERROR disk\nINFO ok\nERROR net\nERROR net\nWARN x\n' > /tmp/phaseb/log.txt && rm -f /tmp/phaseb/errs.txt",
     "prompt": "Read /tmp/phaseb/log.txt. Count how many lines contain the word ERROR and write ONLY that number to /tmp/phaseb/errs.txt.",
     "check": _num("/tmp/phaseb/errs.txt", "3")},

    # --- filter + write ---
    {"id": "filter_lines",
     "setup": r"mkdir -p /tmp/phaseb && printf 'KEEP alpha\ndrop x\nKEEP beta\ndrop y\nKEEP gamma\n' > /tmp/phaseb/src.txt && rm -f /tmp/phaseb/kept.txt",
     "prompt": "Read /tmp/phaseb/src.txt and write ONLY the lines that start with the word KEEP to /tmp/phaseb/kept.txt (preserve them as-is, one per line).",
     "check": r"""test "$(grep -c '^KEEP' /tmp/phaseb/kept.txt 2>/dev/null)" = '3' && ! grep -q 'drop' /tmp/phaseb/kept.txt 2>/dev/null"""},

    {"id": "transform_double",
     "setup": r"mkdir -p /tmp/phaseb && printf '3\n5\n11\n' > /tmp/phaseb/src2.txt && rm -f /tmp/phaseb/doubled.txt",
     "prompt": "Read /tmp/phaseb/src2.txt (one integer per line). Multiply each by 2 and write the results to /tmp/phaseb/doubled.txt, one per line, in the same order.",
     "check": r"""test "$(tr -d '[:space:]' < /tmp/phaseb/doubled.txt 2>/dev/null)" = '61022'"""},

    # --- nested parse ---
    {"id": "json_nested",
     "setup": r"""mkdir -p /tmp/phaseb && printf '{"service": {"db": {"host": "x", "port": 5432}}, "ver": 3}' > /tmp/phaseb/cfg.json && rm -f /tmp/phaseb/nport.txt""",
     "prompt": "Read the JSON file /tmp/phaseb/cfg.json and write ONLY the value of the nested field service.db.port to /tmp/phaseb/nport.txt.",
     "check": _num("/tmp/phaseb/nport.txt", "5432")},

    # --- conditional / branching ---
    {"id": "threshold_branch",
     "setup": r"mkdir -p /tmp/phaseb && printf '40\n35\n30\n' > /tmp/phaseb/t.txt && rm -f /tmp/phaseb/verdict.txt",
     "prompt": "Read /tmp/phaseb/t.txt (integers, one per line). Sum them. If the sum is greater than 100 write exactly HIGH to /tmp/phaseb/verdict.txt, otherwise write exactly LOW.",
     "check": _eq("/tmp/phaseb/verdict.txt", "HIGH")},  # 105 > 100

    # --- multi-file orchestration ---
    {"id": "most_lines",
     "setup": r"mkdir -p /tmp/phaseb && printf 'a\na\n' > /tmp/phaseb/f1.txt && printf 'b\nb\nb\nb\n' > /tmp/phaseb/f2.txt && printf 'c\n' > /tmp/phaseb/f3.txt && rm -f /tmp/phaseb/most.txt",
     "prompt": "Three files exist: /tmp/phaseb/f1.txt, /tmp/phaseb/f2.txt, /tmp/phaseb/f3.txt. Determine which one has the MOST lines and write ONLY that filename (exactly f1.txt, f2.txt, or f3.txt) to /tmp/phaseb/most.txt.",
     "check": r"""grep -q 'f2\.txt' /tmp/phaseb/most.txt 2>/dev/null && ! grep -qE 'f1\.txt|f3\.txt' /tmp/phaseb/most.txt 2>/dev/null"""},

    # --- 3-tool chain ---
    {"id": "chain_hash",
     "setup": r"mkdir -p /tmp/phaseb && printf 'payload' > /tmp/phaseb/word.txt && rm -f /tmp/phaseb/h1.txt /tmp/phaseb/h2.txt",
     "prompt": "Step 1: read the string in /tmp/phaseb/word.txt. Step 2: compute its SHA-256 hex digest and write the first 6 hex chars to /tmp/phaseb/h1.txt. Step 3: read /tmp/phaseb/h1.txt back, convert those 6 chars to UPPERCASE, and write the uppercased version to /tmp/phaseb/h2.txt.",
     "check": r"""H=$(python3 -c "import hashlib;print(hashlib.sha256(b'payload').hexdigest()[:6])"); test "$(tr -dc '0-9a-f' < /tmp/phaseb/h1.txt 2>/dev/null|head -c6)" = "$H" && test "$(tr -dc '0-9A-F' < /tmp/phaseb/h2.txt 2>/dev/null|head -c6)" = "$(echo $H|tr a-f A-F)" """},

    # --- robustness / cleaning ---
    {"id": "clean_sum",
     "setup": r"mkdir -p /tmp/phaseb && printf ' 10 \n1,000\n5\n 20\n' > /tmp/phaseb/dirty.txt && rm -f /tmp/phaseb/cleansum.txt",
     "prompt": "Read /tmp/phaseb/dirty.txt. Each line is an integer but may have surrounding spaces or thousands-separator commas (e.g. 1,000 means 1000). Parse them all as integers, sum them, and write ONLY the total to /tmp/phaseb/cleansum.txt.",
     "check": _num("/tmp/phaseb/cleansum.txt", "1035")},  # 10+1000+5+20
]
