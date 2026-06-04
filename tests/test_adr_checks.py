from tools.adr import check_no_stop_sequences as c1


def test_no_stop_sequences_clean_on_real_file():
    src = c1.TARGET.read_text()
    assert c1.find_violations(src, str(c1.TARGET)) == []


def test_no_stop_sequences_catches_violation():
    bad = "resp = client.chat(messages, max_tokens=4096, stop=['\\ndef '])\n"
    assert c1.find_violations(bad, "fake.py")
