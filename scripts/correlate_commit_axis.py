"""Did the commit axis improve the synthetic board's prediction of real SWE-bench?
For each of the 7 models, read results/<slug>/agentic_native.json (score, score_with_commit,
commit_rate) and results/swebench/<slug>.report.json (resolved/submitted). Compare the
Pearson + Spearman of the PUBLISHED score vs real resolve AND the folded-in score vs real
resolve. Folding in 'earns' a re-publish only if it raises the correlation."""
import json
from pathlib import Path
import numpy as np

SLUGS = ["qwen3-6-27b", "qwen3-5-35b-base", "qwopus-glm-18b", "nemotron-cascade-2-30b",
         "kimi-linear-48b-a3b", "granite-4-1-30b", "nex-n2-mini"]


def pearson(a, b):
    return float(np.corrcoef(np.array(a, float), np.array(b, float))[0, 1])


def spearman(a, b):
    ra = np.argsort(np.argsort(-np.array(a, float))).astype(float)
    rb = np.argsort(np.argsort(-np.array(b, float))).astype(float)
    return pearson(ra, rb)


def main():
    score, folded, commit_rate, resolve, names = [], [], [], [], []
    for s in SLUGS:
        ag = json.loads(Path(f"results/{s}/agentic_native.json").read_text())
        rep = json.loads(Path(f"results/swebench/{s}.report.json").read_text())
        score.append(ag["score"])
        folded.append(ag.get("score_with_commit", ag["score"]))
        commit_rate.append(ag.get("commit_rate", float("nan")))
        resolve.append(100.0 * rep["resolved_instances"] / rep["submitted_instances"])
        names.append(s)
    print(f"{'model':24} {'score':>7} {'folded':>7} {'commit%':>8} {'resolve%':>9}")
    for i, s in enumerate(names):
        print(f"{s:24} {score[i]:7.2f} {folded[i]:7.2f} {commit_rate[i]:8.1f} {resolve[i]:9.1f}")
    print()
    print(f"published score  vs resolve: Pearson {pearson(score, resolve):.3f}  Spearman {spearman(score, resolve):.3f}")
    print(f"folded-in score  vs resolve: Pearson {pearson(folded, resolve):.3f}  Spearman {spearman(folded, resolve):.3f}")

    def rank_of(vals, target):
        order = sorted(range(len(vals)), key=lambda i: -vals[i])
        return order.index(names.index(target)) + 1

    nem = "nemotron-cascade-2-30b"
    print(f"\nNemotron synthetic rank: published #{rank_of(score, nem)} -> folded #{rank_of(folded, nem)}; "
          f"real resolve rank #{rank_of(resolve, nem)} of {len(names)}")


if __name__ == "__main__":
    main()
