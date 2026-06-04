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
