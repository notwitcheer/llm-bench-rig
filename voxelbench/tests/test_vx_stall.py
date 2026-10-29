"""unit test for scripts/vx_stall.py (HARDENING L37): fake state.db + fake /slots server. python3 tests/test_vx_stall.py"""
import json, sqlite3, subprocess, sys, tempfile, threading, time, os
STALL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "vx_stall.py")
from http.server import BaseHTTPRequestHandler, HTTPServer
d = tempfile.mkdtemp(); db = os.path.join(d, "state.db")
c = sqlite3.connect(db); c.execute("create table messages (id integer primary key, timestamp real)"); c.commit()
busy = {"v": False}
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps([{"id": 0, "is_processing": busy["v"]}]).encode())
    def log_message(self, *a): pass
srv = HTTPServer(("127.0.0.1", 0), H); port = srv.server_port
threading.Thread(target=srv.serve_forever, daemon=True).start()
def run(thr, p=port):
    return subprocess.run([sys.executable, STALL, db, str(p), str(thr)], capture_output=True, text=True).stdout.strip()
res = {}
res["no_rows"] = run(60)
c.execute("insert into messages (timestamp) values (?)", (time.time() - 1000,)); c.commit()
res["old_idle_server"] = run(720)      # 1000 s > 720 s, server idle -> stall
res["old_below_thr"] = run(1500)       # 1000 s < 1500 s -> ok
busy["v"] = True
res["old_server_busy"] = run(720)      # long generation -> ok
busy["v"] = False
res["no_server"] = run(720, 1)         # port closed -> unknown
c.execute("insert into messages (timestamp) values (?)", (time.time(),)); c.commit()
res["fresh_row"] = run(720)
res["bad_db"] = subprocess.run([sys.executable, STALL, "/nonexistent.db", str(port), "720"], capture_output=True, text=True).stdout.strip()
print(json.dumps(res, indent=1))
exp = {"no_rows": "unknown", "old_idle_server": "stall", "old_below_thr": "ok", "old_server_busy": "ok", "no_server": "unknown", "fresh_row": "ok", "bad_db": "unknown"}
bad = {k: res[k] for k in exp if not res[k].startswith(exp[k])}
print("UNIT TEST", "PASS" if not bad else f"FAIL {bad}")
