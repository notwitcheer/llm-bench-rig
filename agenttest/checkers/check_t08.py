"""t08 JSON edit: server.port 8443 (int), logging.level "debug", every other key/value unchanged, valid JSON."""
import copy
import json

from _common import main, read, seed_text


def expected():
    d = json.loads(seed_text("t08/config.json"))
    e = copy.deepcopy(d)
    e["server"]["port"] = 8443
    e["logging"]["level"] = "debug"
    return e


def check(work):
    s = read(work, "t08/config.json")
    if s is None:
        return False, "config.json missing"
    try:
        d = json.loads(s)
    except Exception as ex:  # noqa: BLE001
        return False, f"invalid JSON: {ex}"
    if d != expected():
        return False, f"content differs: {json.dumps(d)[:200]}"
    if type(d["server"]["port"]) is not int:
        return False, "port is not an int"
    return True, "port 8443, level debug, rest unchanged"


if __name__ == "__main__":
    main(check)
