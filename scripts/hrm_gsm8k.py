"""t047 HRM-Text-1B GSM8K-generative spot-check (authors claim 84.5).
Correct protocol = condition tokens + token_type_ids PrefixLM mask; run again
with --no-prefix-mask to MEASURE the silent harness trap. Raw gens captured
to JSONL for offline re-scoring (rig rule: absurd score = check the harness).

  cd ~/benchmark-rig && ~/hrm-env/bin/python scripts/hrm_gsm8k.py --n 200
  cd ~/benchmark-rig && ~/hrm-env/bin/python scripts/hrm_gsm8k.py --n 200 --no-prefix-mask
"""
import argparse, json, random, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from scripts.train.gsm8k_extract import extract_gsm8k_answer

MODEL = "sapientinc/HRM-Text-1B"
COND = {"direct": "<|object_ref_start|>", "cot": "<|object_ref_end|>",
        "noisy": "<|quad_start|>", "synth": "<|quad_end|>",
        "synth+cot": "<|quad_end|><|object_ref_end|>"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cond", default="synth+cot", choices=list(COND))
    ap.add_argument("--no-prefix-mask", action="store_true")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset
    test = load_dataset("openai/gsm8k", "main", split="test")
    idxs = random.Random(args.seed).sample(range(len(test)), args.n)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    tag = f"{args.cond}{'-nomask' if args.no_prefix_mask else ''}-n{args.n}"
    out_path = f"results/hrm-gsm8k-{tag}.jsonl"
    correct, t0 = 0, time.time()
    with open(out_path, "w") as f:
        for i, idx in enumerate(idxs):
            item = test[idx]
            gold = extract_gsm8k_answer(item["answer"])
            prompt = f"<|im_start|>{COND[args.cond]}{item['question']}<|im_end|>"
            inputs = tok(prompt, return_tensors="pt").to("cuda")
            if not args.no_prefix_mask:
                inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new,
                                     do_sample=False)
            gen = tok.decode(out[0, inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            pred = extract_gsm8k_answer(gen)
            ok = pred is not None and pred == gold
            correct += ok
            f.write(json.dumps({"idx": idx, "gold": gold, "pred": pred,
                                "ok": ok, "gen": gen}) + "\n")
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{args.n}  acc={correct/(i+1):.3f}  "
                      f"({(time.time()-t0)/(i+1):.1f}s/q)", flush=True)

    print(json.dumps({"tag": tag, "n": args.n, "acc": round(correct / args.n, 4),
                      "raw": out_path, "minutes": round((time.time() - t0) / 60, 1)}))


if __name__ == "__main__":
    main()
