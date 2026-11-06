# astra-stop Q4 replay readout

VERDICT REPRODUCED | median thinking tokens -35.5%, p=0.0000
LINE1 on Q4_K_M the edited Qwen3.8-27B thought a median 16930 tokens on the pelican prompt vs 26254 unedited (-36% median, -46% mean, Mann-Whitney p=7.54e-06, n=32/32); SVGs complete 97% vs 100%.
{

regime: llama.cpp fac3a894f, Q4_K_M both arms, KV q8_0, ctx 262144/4 slots, sampler temp 1.0 top-p 0.95 top-k 20, reasoning effort xhigh, seeds 1-32, cap 60000
