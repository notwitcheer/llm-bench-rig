"""Phase B task battery — real agent tasks for Hermes Agent.

Each task: setup (shell, creates fixtures under /tmp/phaseb), prompt (given to
`hermes -z`), check (shell that returns 0 iff the resulting filesystem state is
correct). Local + deterministic; success is an objective filesystem assertion,
not output parsing — robust to the model's prose. Raw strings keep `\\n` literal
for printf.
"""

TASKS = [
    {
        "id": "sum_csv",
        "setup": r"mkdir -p /tmp/phaseb && printf '12\n30\n8\n50\n' > /tmp/phaseb/input.csv && rm -f /tmp/phaseb/out.txt",
        "prompt": "Read the file /tmp/phaseb/input.csv (it has one integer per line). Sum the integers and write ONLY the total number to /tmp/phaseb/out.txt. Use your tools to do it.",
        "check": r"""test "$(tr -dc '0-9' < /tmp/phaseb/out.txt 2>/dev/null)" = '100'""",
    },
    {
        "id": "sha_prefix",
        "setup": r"mkdir -p /tmp/phaseb && rm -f /tmp/phaseb/hash.txt",
        "prompt": "Compute the SHA-256 hex digest of the exact ASCII string hermes-agent and write ONLY the first 8 hex characters of that digest to /tmp/phaseb/hash.txt. Use your tools to compute it.",
        "check": r"""test "$(tr -dc '0-9a-f' < /tmp/phaseb/hash.txt 2>/dev/null | head -c 8)" = "$(python3 -c "import hashlib;print(hashlib.sha256(b'hermes-agent').hexdigest()[:8])")" """,
    },
    {
        "id": "uppercase",
        "setup": r"mkdir -p /tmp/phaseb && printf 'hello hermes world\n' > /tmp/phaseb/notes.txt && rm -f /tmp/phaseb/upper.txt",
        "prompt": "Read /tmp/phaseb/notes.txt, convert all of its text to UPPERCASE, and write the uppercased text to /tmp/phaseb/upper.txt. Use your tools.",
        "check": r"""grep -q 'HELLO HERMES WORLD' /tmp/phaseb/upper.txt 2>/dev/null""",
    },
    {
        "id": "count_py",
        "setup": r"mkdir -p /tmp/phaseb/code && rm -f /tmp/phaseb/code/* /tmp/phaseb/count.txt && touch /tmp/phaseb/code/a.py /tmp/phaseb/code/b.py /tmp/phaseb/code/c.py /tmp/phaseb/code/readme.md",
        "prompt": "Count how many files ending in .py are in the directory /tmp/phaseb/code, and write ONLY that number to /tmp/phaseb/count.txt. Use your tools.",
        "check": r"""test "$(tr -dc '0-9' < /tmp/phaseb/count.txt 2>/dev/null)" = '3'""",
    },
    {
        "id": "bigger_file",
        "setup": r"mkdir -p /tmp/phaseb && printf 'a\na\na\na\na\n' > /tmp/phaseb/a.txt && printf 'b\nb\n' > /tmp/phaseb/b.txt && rm -f /tmp/phaseb/bigger.txt",
        "prompt": "Compare the two files /tmp/phaseb/a.txt and /tmp/phaseb/b.txt by their number of lines. Write ONLY the filename (exactly a.txt or b.txt) of whichever has MORE lines to /tmp/phaseb/bigger.txt. Use your tools.",
        "check": r"""grep -q 'a\.txt' /tmp/phaseb/bigger.txt 2>/dev/null && ! grep -q 'b\.txt' /tmp/phaseb/bigger.txt 2>/dev/null""",
    },
    {
        "id": "extract_json",
        "setup": r"""mkdir -p /tmp/phaseb && printf '{"host": "local", "port": 8090, "name": "capsule"}' > /tmp/phaseb/data.json && rm -f /tmp/phaseb/port.txt""",
        "prompt": "Read the JSON file /tmp/phaseb/data.json and write ONLY the value of its 'port' field to /tmp/phaseb/port.txt. Use your tools.",
        "check": r"""test "$(tr -dc '0-9' < /tmp/phaseb/port.txt 2>/dev/null)" = '8090'""",
    },
]
