"""Embedding model lineup for the t037 retrieval bench. One config per arm.

Prompt conventions matter for quality:
- e5 family: asymmetric "query: " / "passage: " prefixes (mandatory).
- Qwen3-Embedding / Qwen3-VL-Embedding: an instruction prefix on the QUERY side only;
  documents are embedded raw. Instruction text follows the Qwen model card.
"""

QWEN_QUERY_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)

MODELS = {
    # batch_size is tuned to co-reside with Donald (~18.9GB VRAM held by llama-server),
    # leaving ~12GB for the encode. Bigger models -> smaller batch.
    "e5-small": {
        "hf_id": "intfloat/e5-small-v2",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "native_dim": 384,
        "batch_size": 64,
    },
    "qwen3-text": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "query_prefix": QWEN_QUERY_INSTRUCT,
        "doc_prefix": "",
        "native_dim": 1024,
        "batch_size": 16,
    },
    "qwen3-vl": {
        # official repo (the tomaarsen vdr variant ships no recognized image_processor_type);
        # needs pillow + torchvision in the venv, NOT a CUDA toolkit (unlike t036's vLLM wall).
        "hf_id": "Qwen/Qwen3-VL-Embedding-2B",
        "query_prefix": QWEN_QUERY_INSTRUCT,
        "doc_prefix": "",
        "native_dim": 2048,
        "batch_size": 8,
    },
}

ARMS = ["e5-small", "qwen3-text", "qwen3-vl"]
