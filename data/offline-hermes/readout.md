# offline Hermes readout

line 1 (as it would post, finding-first): Hermes Agent with the network unplugged: 9 of 10 everyday tasks pass with zero outbound packets (the 10th, a web fetch, cannot work offline; 9 of the 9 offline-capable tasks)

status: COMPLETE; isolation proof pre PASS, post PASS; non-loopback packets after the run: 0

| task | what | pass | wall s | calls | tool errs | approval blocks | check reason |
|---|---|---|---:|---:|---:|---:|---|
| t01 | create a file | PASS | 14.2 | 2 | 0 | 0 | exact single line |
| t02 | CSV column total | PASS | 30.9 | 9 | 1 | 4 | 6593.60 == 6593.60 |
| t03 | find a phrase in 5 files | PASS | 7.2 | 3 | 0 | 0 | note_douglas.txt |
| t04 | fix a bug until tests pass | PASS | 6.8 | 4 | 0 | 0 | test.py + hidden test pass, test.py unchanged |
| t05 | folder + 3 files + index | PASS | 6.7 | 4 | 0 | 0 | 3 files + sorted index |
| t06 | fetch a web page, write a fact | FAIL | 124.5 | 22 | 4 | 3 | 'Hermes Agent' |
| t07 | count ERROR lines in a log | PASS | 12.4 | 2 | 0 | 0 | 32 == 32 |
| t08 | edit two JSON keys | PASS | 30.2 | 10 | 1 | 4 | port 8443, level debug, rest unchanged |
| t09 | recover from a failing script | PASS | 9.3 | 4 | 0 | 0 | report produced, script unchanged |
| t10 | 40 lines to 3 bullets | PASS | 20.5 | 4 | 0 | 0 | 3 bullets with date, budget, owner |

totals: 64 tool calls, 6 other tool errors, 11 approval blocks, 0 malformed, 0 timeouts, wall 262.7 s.

## what offline Hermes cannot do: t06 (fetch a web page)

- the task: fetch https://hermes-agent.nousresearch.com/docs/ and write the product name. there is no route out, so no fetch can succeed (successful fetches recorded: 3).
- what the agent did: 7 network attempts; class `ALARM_fetch_succeeded_offline` (final text admits failure: True; claims a fetch: False; wrote fact.txt: True; the v1 checker alone would say True).
- scoring rule fixed before the run: t06 is NOT passed offline whatever the checker says. a model can write `Hermes Agent` from its own prompt, which the checker accepts; the page was never read. the class heuristic is regex-based, read `t06.jsonl` before quoting it.

## the angle (every readout ends with it)

- line 1: Hermes Agent with the network unplugged: 9 of 10 everyday tasks pass with zero outbound packets (the 10th, a web fetch, cannot work offline; 9 of the 9 offline-capable tasks).
- rule used: gate rule 4, AI at home / fully local Hermes piece, reproducible (unplugged-cable frame: `docker inspect` NetworkMode none + a failing curl, then the agent finishing tasks).
- closest comparables [GATE, IMP]: fourth Hermes Recipe fully local 26,073; run AI at home 22,334 / 56,603; bracket 6-26k.
- call: X only with the proof frame as the picture and an honest t06 line; if fewer than 8 of the 9 offline-capable tasks pass, say that plainly or hold for TG + repo.
- limits: our own 10 tasks (agenttest v1, saturated by good models online: Spark-X2.6-4B 10/10, Muse Glimmer 10/10), one sampled run, n=10, not a benchmark; the model server is on the host and reached through a unix socket, so `offline` means the AGENT has no network, not that the machine has none.
