"""Fixed prompt set for the served (http) speed lane.

Four workloads, eight prompts each, sent verbatim at temperature 0 so any two
runs of the same server flags produce the same text (checked by sha256 in
lib/speed_served.py). WORKLOADS is copied byte for byte from the
spec_cache_bench.py client that produced the speculative-decoding and
prompt-cache reports; do not edit the prompts, add a new dict instead, or
every earlier jsonl record stops being comparable.
"""

WORKLOADS = {
    "prose": [
        "Write a detailed essay about the history of the printing press.",
        "Describe a walk through a forest in autumn, in rich detail.",
        "Explain the causes of the First World War for a general reader.",
        "Write a biographical sketch of Marie Curie.",
        "Describe the water cycle in flowing prose.",
        "Write an opinion piece about public libraries.",
        "Narrate the story of a lighthouse keeper's ordinary day.",
        "Explain how photosynthesis works, in essay form.",
    ],
    "code": [
        "Write a Python function that merges two sorted lists.",
        "Implement binary search in Python with comments.",
        "Write a Python class for a simple LRU cache.",
        "Write a function that validates an email address with a regex in Python.",
        "Implement quicksort in Python.",
        "Write a Python script that counts word frequencies in a text file.",
        "Implement a linked list with insert and delete in Python.",
        "Write a Python function to compute the nth Fibonacci number iteratively.",
    ],
    "repetitive": [
        "List the numbers from 1 to 100, one per line.",
        "Print a 12x12 multiplication table.",
        "List the days of the week repeated 20 times.",
        "Write the alphabet 15 times, one repetition per line.",
        "Generate a CSV with columns id,name,value for 60 rows of dummy data.",
        "List every month of the year ten times.",
        "Produce a markdown table of 50 rows numbering items Item 1 to Item 50.",
        "Repeat the sentence 'the quick brown fox jumps over the lazy dog' 30 times, numbered.",
    ],
    "chat": [
        "What's a good way to learn to cook?",
        "How do I stay motivated to exercise?",
        "What should I consider when adopting a dog?",
        "Any tips for a first trip to Japan?",
        "How do I make my mornings less chaotic?",
        "What's a sensible way to start investing small amounts?",
        "How can I get better at small talk?",
        "What are good habits for improving sleep?",
    ],
}


WORKLOAD_NAMES = tuple(WORKLOADS)
