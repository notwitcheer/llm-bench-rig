from lib.agentic.multistep import detect_derailment

def test_derailment_on_repeat():
    actions = ["result = read_file('/x')", "result = read_file('/x')", "result = read_file('/x')"]
    assert detect_derailment(actions) is True

def test_no_derailment_on_progress():
    actions = ["result = read_file('/x')", "result = web_search('y')"]
    assert detect_derailment(actions) is False
