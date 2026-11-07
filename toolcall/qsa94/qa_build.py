#!/usr/bin/env python3
"""qa_build.py <corpus.txt> <outdir>: builds the item-94 long-context set from the fetched Hermes docs text (deterministic; the item's PLAN phase re-runs it into a temp dir and compares hashes).
outputs: docs_slice.txt (the first DOC_CHARS characters of the corpus, cut at a line boundary), prefill.json (the two Hermes prefill messages that carry the slice), questions.json (id, depth fraction, question text),
keys/keys.json (answer keys: harness folder only, never inside the agent's work dir; the agent's toolset has no file or terminal tool).
A question = 'what is the default of `<key>`' for a YAML-style line `key: <number>   # comment` in the docs whose key appears exactly once in the slice, so the answer is unique."""
import json, os, re, sys

DOC_CHARS = 985_000
NQ = 24
corpus, out = sys.argv[1], sys.argv[2]
t = open(corpus, encoding="utf-8").read()
sl = t[:DOC_CHARS]
sl = sl[:sl.rfind("\n") + 1]
os.makedirs(out + "/keys", exist_ok=True)
open(out + "/docs_slice.txt", "w", encoding="utf-8").write(sl)
srcs = [(m.start(), m.group(1)) for m in re.finditer(r"<!-- source: (.*?) -->", sl)]


def src_at(p):
    s = None
    for a, b in srcs:
        if a <= p:
            s = b
        else:
            break
    return s


cands = []
for m in re.finditer(r"^\s{0,8}([a-z][a-z_]{7,40}): (\d[\d_.]*)\s+#\s*(.{12,100})$", sl, re.M):
    k, v = m.group(1), m.group(2)
    if len(re.findall(r"(?<![\w])" + re.escape(k) + r"(?![\w])", sl)) != 1:
        continue
    cands.append((m.start(), k, v, m.group(3).strip()))
for m in re.finditer(r"`([a-z][a-z_.]{7,40})` \(default `?(\d[\d_.]*)`?[,)]", sl):
    k, v = m.group(1), m.group(2)
    if len(re.findall(r"(?<![\w])" + re.escape(k) + r"(?![\w])", sl)) == 1:
        cands.append((m.start(), k, v, "prose"))
cands.sort()
# NQ facts spread over the slice: the candidate nearest to each of NQ evenly spaced positions (no repeats)
pick, used = [], set()
for i in range(NQ):
    target = (i + 0.5) * len(sl) / NQ
    best = min((c for c in cands if c[1] not in used), key=lambda c: abs(c[0] - target), default=None)
    if best:
        used.add(best[1])
        pick.append(best)
pick.sort()
qs, keys = [], {}
for i, (p, k, v, c) in enumerate(pick, 1):
    qid = "Q%02d" % i
    qs.append({"id": qid, "depth": round(p / len(sl), 3), "q": "In the Hermes Agent documentation above, what is the default value of `%s`? Answer with the number only." % k})
    keys[qid] = {"key": k, "value": v, "source": src_at(p), "pos": p}
docs_msg = ("Below is the complete Hermes Agent documentation text. Answer questions about it from this text only, as briefly as asked. Do not guess a value that is not written here.\n\n" + sl)
prefill = [{"role": "user", "content": docs_msg}, {"role": "assistant", "content": "I have the documentation text and will answer from it only."}]
json.dump(prefill, open(out + "/prefill.json", "w"))
json.dump(qs, open(out + "/questions.json", "w"), indent=1)
json.dump(keys, open(out + "/keys/keys.json", "w"), indent=1)
print("slice chars", len(sl), "questions", len(qs), "depth span %.2f-%.2f" % (qs[0]["depth"], qs[-1]["depth"]) if qs else "none")
