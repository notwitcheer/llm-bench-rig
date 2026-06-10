"""Word-level WER = (substitutions + deletions + insertions) / reference_words, via a
classic Levenshtein DP over token lists. Dep-free (no jiwer). Text is normalized first
so casing/punctuation never counts as an error."""
from lib.asr.normalize import normalize


def _edit_distance(ref_words: list, hyp_words: list) -> int:
    m, n = len(ref_words), len(hyp_words)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = prev if ref_words[i - 1] == hyp_words[j - 1] else 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    return dp[n]


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)
