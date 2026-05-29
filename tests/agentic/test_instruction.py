from lib.agentic.instruction import check_compliance

def test_exact():
    assert check_compliance("DONE", {"type": "exact", "value": "DONE"}) is True
    assert check_compliance("done.", {"type": "exact", "value": "DONE"}) is False

def test_regex():
    assert check_compliance("red,green,blue", {"type": "regex", "value": "^[a-z]+,[a-z]+,[a-z]+$"}) is True

def test_json_keys():
    assert check_compliance('{"ok": true}', {"type": "json_keys", "value": ["ok"]}) is True

def test_forbids():
    assert check_compliance("here you go", {"type": "forbids", "value": "sorry"}) is True
    assert check_compliance("sorry, no", {"type": "forbids", "value": "sorry"}) is False
