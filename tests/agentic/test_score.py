from lib.agentic.score import pairing_score

WEIGHTS = {"codeact": 0.4, "longcontext": 0.25, "instruction": 0.2, "multistep": 0.15}

def test_weighted_sum():
    evals = {"codeact": {"score": 80}, "longcontext": {"score": 60},
             "instruction": {"score": 90}, "multistep": {"score": 50}}
    # 0.4*80 + 0.25*60 + 0.2*90 + 0.15*50 = 32 + 15 + 18 + 7.5 = 72.5
    assert pairing_score(evals, WEIGHTS) == 72.5

def test_missing_eval_scores_zero():
    evals = {"codeact": {"score": 100}}
    assert pairing_score(evals, WEIGHTS) == 40.0
