from tools.adr import check_no_stop_sequences as c1
from tools.adr import check_think_recorded as c2


def test_no_stop_sequences_clean_on_real_file():
    src = c1.TARGET.read_text()
    assert c1.find_violations(src, str(c1.TARGET)) == []


def test_no_stop_sequences_catches_violation():
    bad = "resp = client.chat(messages, max_tokens=4096, stop=['\\ndef '])\n"
    assert c1.find_violations(bad, "fake.py")


def test_think_recorded_clean_on_real_file():
    src = c2.TARGET.read_text()
    assert c2.find_violations(src, str(c2.TARGET)) == []


def test_think_recorded_catches_missing():
    bad = (
        "def run_benchmark(model_path):\n"
        "    meta = extract_metadata(model_path)\n"
        "    out_dir = save_metadata(meta, results_dir)\n"
    )
    assert c2.find_violations(bad, "fake.py")


def test_think_recorded_catches_wrong_order():
    bad = (
        "def run_benchmark(model_path):\n"
        "    meta = extract_metadata(model_path)\n"
        "    out_dir = save_metadata(meta, results_dir)\n"
        "    meta['think'] = True\n"
    )
    assert c2.find_violations(bad, "fake.py")


def test_think_recorded_catches_missing_save():
    bad = (
        "def run_benchmark(model_path):\n"
        "    meta = extract_metadata(model_path)\n"
        "    meta['think'] = True\n"
    )
    assert c2.find_violations(bad, "fake.py")


from tools.adr import run_all


def test_run_all_clean_tree_returns_zero():
    assert run_all.main() == 0


from tools.adr import check_mc_gate_wired as c3


def test_mc_gate_wired_clean_on_real_files():
    for target in c3.TARGETS:
        assert c3.find_violations(target.read_text(), str(target)) == []


def test_mc_gate_wired_catches_unwired_eval():
    bad = (
        "class FakeEval:\n"
        "    def evaluate(self):\n"
        "        response = self.client.chat(messages)\n"
        "        predicted = parse_choice(response)\n"
    )
    assert c3.find_violations(bad, "fake.py")
