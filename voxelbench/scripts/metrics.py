#!/usr/bin/env python3
"""voxel bench metrics from a bench profile's state.db. usage: metrics.py <profile_dir> <out_json>"""
import json, sqlite3, sys, re
from pathlib import Path

prof = Path(sys.argv[1]); out = Path(sys.argv[2])
c = sqlite3.connect(f"file:{prof / 'state.db'}?mode=ro", uri=True)
c.row_factory = sqlite3.Row
S = [dict(r) for r in c.execute(
    "select id, source, parent_session_id, model, started_at, ended_at, end_reason, message_count, tool_call_count, "
    "input_tokens, output_tokens, reasoning_tokens, api_call_count, compression_ineffective_count from sessions order by started_at")]
tool_err = 0; tools = {}
for r in c.execute("select tool_name, content from messages where role='tool'"):
    tools[r["tool_name"] or "?"] = tools.get(r["tool_name"] or "?", 0) + 1
    t = (r["content"] or "")[:4000]
    if re.search(r'"(error|success)"\s*:\s*(?:"[^"]+"|false)', t) and ('"error"' in t or '"success": false' in t):
        tool_err += 1
compacted = c.execute("select count(*) from messages where compacted=1").fetchone()[0]
goals = [json.loads(v) for (v,) in c.execute("select value from state_meta where key like 'goal:%'")]
g = goals[-1] if goals else {}
res = {
    "sessions": len(S),
    "child_sessions": sum(1 for s in S if s["parent_session_id"]),
    "messages": sum(s["message_count"] or 0 for s in S),
    "tool_calls": sum(s["tool_call_count"] or 0 for s in S),
    "api_calls": sum(s["api_call_count"] or 0 for s in S),
    "input_tokens": sum(s["input_tokens"] or 0 for s in S),
    "output_tokens": sum(s["output_tokens"] or 0 for s in S),
    "tool_results_by_name": tools,
    "tool_errors_flagged": tool_err,
    "compacted_messages": compacted,
    "goal": {k: g.get(k) for k in ("status", "turns_used", "max_turns", "last_reason", "paused_reason") if k in g},
    "session_rows": [{k: s[k] for k in ("id", "source", "parent_session_id", "end_reason", "message_count", "tool_call_count",
                                         "input_tokens", "output_tokens")} for s in S],
}
out.write_text(json.dumps(res, indent=2))
print(json.dumps({k: res[k] for k in ("sessions", "child_sessions", "tool_calls", "api_calls", "input_tokens",
                                      "output_tokens", "tool_errors_flagged", "compacted_messages", "goal")}))
