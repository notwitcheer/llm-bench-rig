# agenttest/offline: Hermes Agent with the network cable pulled

Hermes Agent runs inside `docker run --network none` and does 10 everyday tasks against a local model. The model server lives on the host (loopback only) and is
reached through a unix socket. The point is a reproducible, checkable claim: the agent finishes tasks while it provably has no network.

## shape

```
host                                   container (--network none, user 1000, cap-drop ALL)
llama-server 127.0.0.1:PORT            Hermes Agent (official image, pinned id)
  ^                                      -> http://127.0.0.1:18080/v1   (lo, inside the container only)
bridge_host.py  <== unix socket ==>    bridge_ctr.py (PID 1): 127.0.0.1:18080 -> /bridge/model.sock
(run dir)/bridge/model.sock  (bind mount, read-only)
```

- no socat on the box and none needed: both relays are ~60 lines of stdlib asyncio.
- the container sees only: its Hermes home, the task work dir (mounted at the path the frozen v1 prompts name), the socket directory (ro), the relay script (ro).
  It never sees the checkers, the seeds or the hand solutions: the v1 checkers run in a separate throwaway `--network none` container after each task.
- image: `nousresearch/hermes-agent:v0.22.4` (2026.9.24, upstream 4d043038), pinned by image id in the item. Why not the python:3.12-slim + installer recipe: that install needs
  network at build time and tracks a moving ref; the official image is already on the box, pinned, ships curl + python, and runs with `--entrypoint` straight to `hermes`.
  (the recipe in `agent-harness-evals.md` stays the fallback: `libatomic1`, installer from the same ref, `hermes --version`.)

## the proof (isolation.py, recorded raw in `<run>/isolation/`)

| leg | what | pass condition |
|---|---|---|
| L1 | `docker inspect` | NetworkMode `none`, only network `none`, no ports, no IP |
| L2 | probes inside: curl https://example.com, curl 1.1.1.1, curl the docker0 gateway :8090 (Donald), getent hosts, python socket connect, /proc/net/route | every probe fails |
| L3 | `/sys/class/net/*/statistics` inside, after the probes and again after the tasks | zero non-loopback interfaces, zero non-loopback packets. `lo` carries the Hermes -> relay traffic and is reported separately |
| C0 | control: same image + flags, no bridge, idle + probes | every counter, `lo` included, reads zero |
| C1 | positive control: same curl on the default network (skipped in DRYRUN) | succeeds with eth0 packets, so the failing curl is the flag and not a broken tool |

The proof runs before the model server starts; a FAIL stops the item with rc 5 and the card is never touched. A forced default network (`OFF_REDPATH_NETWORK=bridge`, test knob)
makes L1/L2/L3 fail, which was verified.

## tasks and scoring

The 10 frozen agenttest v1 tasks (`../tasks.json`, prompts verbatim, one fresh session each). t06 fetches a web page and cannot work offline: it is scored NOT passed whatever the v1 checker
says (a model can write "Hermes Agent" from memory and the checker accepts it), and its transcript is classified (honest_no_fetch / silent_guess_from_memory /
claimed_a_fetch_that_did_not_happen / gave_up_silently; regex heuristic, read `t06.jsonl`). Line 1 counts passes of 10; the readout also gives passes of the 9 offline-capable tasks.
Hermes is run as shipped: default toolsets, default approvals, memory off, telemetry off, no sampling fields sent (the server flags carry the maker-card sampling, checked against `/props`).

## files

- `offline_run.py` driver: `up` (seed, bridge, container, proof), `tasks` (resumable, deadline + card-cap aware), `down` (post counters, readout, cleanup), `proof`, `equivalence`
- `isolation.py` the proof legs; `bridge_host.py`, `bridge_ctr.py` the relays; `seedlib.py` seeds the v1 tree with the container work path (`equivalence` proves it equals `gen_seeds.build()`)
- `mock_offline.py` scripted OpenAI-compatible mock that solves the 10 tasks through real Hermes tool calls (PASS path without a model); `MOCK_T06=guess`, `MOCK_FAIL=tNN,..`, `MOCK_HANG=s` modes
- `freeze_offline.py` -> `FROZEN.sha256` (the item checks it at every DRYRUN and start)
- queue item: `scripts/capsule-queue/75-offline-hermes.sh` (night-lib contract, Donald drain + EXIT-trap restore, exit 3 resumable, refuses under 35 min left)

## honest limits

n=10 on our own tasks (v1 is saturated online: Spark-X2.6-4B 10/10, Muse Glimmer 10/10), one sampled run, not a benchmark. "Offline" = the agent has no network; the host has one and serves the model.
Bytes on `lo` are the model traffic, not egress. No GPU speed claim.
