"""Materialize 3 contrasting single-stream workloads (30 prompts each) spanning the
n-gram-repetition spectrum. Run ONCE on capsule (needs `datasets`); output committed for
reproducibility.
  code     -> HumanEval prompts             (high structural repetition)
  chat     -> open-ended instructions       (low repetition, no copyable context)
  copyctx  -> summarize-this-document        (output copies input spans; PLD sweet spot)

Usage (capsule, in the cacheback env):
  ~/cacheback-env/bin/python scripts/build_cacheback_workloads.py
"""
import json
import os

from datasets import load_dataset

os.makedirs("dataset/cacheback", exist_ok=True)


def dump(wl, prompts):
    assert len(prompts) == 30, f"{wl}: expected 30 prompts, got {len(prompts)}"
    with open(f"dataset/cacheback/{wl}.jsonl", "w") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"id": f"{wl}-{i:02d}", "workload": wl, "prompt": p}) + "\n")


# --- code: first 30 HumanEval prompts (signature + docstring -> complete it) ---
he = load_dataset("openai/openai_humaneval", split="test")
dump("code", [he[i]["prompt"] for i in range(30)])

# --- chat: 30 fixed, diverse, single-turn prompts with NO long copyable context ---
chat = [
    "Explain why the sky is blue to a curious ten-year-old.",
    "What are the main trade-offs between renting and buying a home?",
    "Describe how a bill becomes law in a parliamentary democracy.",
    "Give three practical tips for sleeping better at night.",
    "What is the difference between weather and climate?",
    "Suggest a simple weeknight dinner I can cook in under 30 minutes.",
    "Explain the concept of compound interest and why it matters.",
    "What causes the seasons to change throughout the year?",
    "How would you explain machine learning to someone with no technical background?",
    "Write a short, encouraging note to a friend who failed an exam.",
    "What are some effective ways to reduce stress during a busy week?",
    "Explain the difference between a virus and a bacterium.",
    "Describe the water cycle in a few clear sentences.",
    "What should a beginner consider before adopting a dog?",
    "Give a balanced view on the pros and cons of remote work.",
    "How does a refrigerator keep food cold?",
    "Recommend three classic novels and briefly say why each is worth reading.",
    "What is the greenhouse effect and how does it work?",
    "Explain how vaccines help the immune system.",
    "Offer advice to someone starting their first job interview tomorrow.",
    "What are the key differences between coffee and tea, beyond caffeine?",
    "Describe how tides are influenced by the moon.",
    "Suggest a beginner-friendly exercise routine for someone who sits all day.",
    "Explain what inflation is and how it affects everyday spending.",
    "Why do leaves change color in autumn?",
    "Give three tips for writing a clear and concise email.",
    "How does the internet send a message from one computer to another?",
    "What is the difference between empathy and sympathy?",
    "Describe a simple way to start meditating for the first time.",
    "Explain why exercise is good for mental health, not just physical health.",
]
dump("chat", chat)

# --- copyctx: 30 CNN/DailyMail articles, truncated, with a summarize instruction ---
cd = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
copyctx = [
    f"Summarize the following article in three sentences.\n\nARTICLE:\n{cd[i]['article'][:3000]}\n\nSUMMARY:"
    for i in range(30)
]
dump("copyctx", copyctx)

print("wrote dataset/cacheback/{code,chat,copyctx}.jsonl")
