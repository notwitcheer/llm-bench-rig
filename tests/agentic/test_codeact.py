from lib.agentic.codeact import check_result

def test_equals_int():
    assert check_result(46, {"type": "equals", "value": 46}) is True

def test_equals_str_casts():
    assert check_result("32GB", {"type": "equals", "value": "32GB"}) is True

def test_startswith():
    assert check_result("sent[log]:42", {"type": "startswith", "value": "sent[log]:"}) is True

def test_contains_false():
    assert check_result("nope", {"type": "contains", "value": "yes"}) is False

def test_approx():
    assert check_result(3.14159, {"type": "approx", "value": 3.14, "tol": 0.01}) is True
