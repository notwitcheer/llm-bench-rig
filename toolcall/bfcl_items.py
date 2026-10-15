"""BFCL v4 glue for toolcall-v1: load items, build tools, call BFCL's UNCHANGED checkers.

Pinned: bfcl-eval 2026.3.23 (PyPI) == gorilla commit 6ea57973 (berkeley-function-call-leaderboard),
in its own uv venv on capsule (~/toolcall-v1/.venv), never the rig venv (numpy==1.26.4 pin + vendor SDKs).

Tools are built the way BFCL builds them for OpenAI-compatible FC models:
  bfcl_eval.utils.load_dataset_entry (adds the "Note that the provided function is in Python 3 syntax." hint)
  -> bfcl_eval.model_handler.utils.convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
which maps BFCL types to JSON-schema types and replaces "." in names with "_". The checker undoes the
name change through the model config's underscore_to_dot flag, so we register one runtime-only model
config (no site-packages patch) with underscore_to_dot=True.
"""
from __future__ import annotations

import copy
from functools import lru_cache

from bfcl_eval.constants.enums import Language, ModelStyle
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.utils import (is_empty_output, is_function_calling_format_output, load_dataset_entry,
                             load_ground_truth_entry)

CHECKER_MODEL = "toolcall-raw-qwen38-xml"   # no "_" in it: BFCL maps "_" -> "/" before the registry lookup
MODEL_CONFIG_MAPPING.setdefault(CHECKER_MODEL, ModelConfig(
    model_name=CHECKER_MODEL, display_name="toolcall-v1 raw-mode Qwen3.8 XML (local)", url="", org="local",
    license="", model_handler=QwenFCHandler, is_fc_model=True, underscore_to_dot=True))

SINGLE_CATS = ["simple_python", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple",
               "live_parallel", "live_parallel_multiple", "irrelevance"]


@lru_cache(maxsize=None)
def _entries(cat: str) -> dict:
    return {e["id"]: e for e in load_dataset_entry(cat)}


@lru_cache(maxsize=None)
def _gt(cat: str) -> dict:
    if cat == "irrelevance":
        return {}
    return {e["id"]: e for e in load_ground_truth_entry(cat)}


def category_of(item_id: str) -> str:
    return item_id.rsplit("_", 1)[0]


def entry(item_id: str) -> dict:
    return copy.deepcopy(_entries(category_of(item_id))[item_id])


def ground_truth(item_id: str):
    cat = category_of(item_id)
    return copy.deepcopy(_gt(cat)[item_id]["ground_truth"]) if _gt(cat) else None


def all_ids(cat: str) -> list[str]:
    return list(_entries(cat).keys())


def tools_for(e: dict) -> list[dict]:
    return convert_to_tool(e["function"], GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)


def messages_for(e: dict) -> list[dict]:
    """single-turn: question[0] is the first (only) turn's message list. FC mode adds no system prompt
    (QwenFCHandler._pre_query_processing_prompting: 'FC models use its own system prompt')."""
    return copy.deepcopy(e["question"][0])


def check_single(item_id: str, e: dict, decoded: list[dict] | None, decode_error: str | None) -> dict:
    """Mirror of bfcl eval_runner._evaluate_single_ast_entry / _evaluate_single_relevance_entry with
    our decode in place of handler.decode_ast. Returns {"valid": bool, "error_type": str|None, ...}."""
    cat = category_of(item_id)
    if cat == "irrelevance":
        contain = decoded is not None and not is_empty_output(decoded)
        return {"valid": not contain,
                "error_type": None if not contain else "irrelevance_error:decoder_success"}
    if decoded is None:
        return {"valid": False, "error_type": "ast_decoder:decoder_failed", "error": [decode_error or ""]}
    if not is_function_calling_format_output(decoded):
        return {"valid": False, "error_type": "ast_decoder:decoder_wrong_output_format"}
    r = ast_checker(e["function"], decoded, ground_truth(item_id), Language.PYTHON, cat, CHECKER_MODEL)
    return {"valid": bool(r["valid"]), "error_type": r.get("error_type"), "error": r.get("error")}


_PY_OK = {"integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
          "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
          "string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
          "array": lambda v: isinstance(v, list), "object": lambda v: isinstance(v, dict)}


def schema_violations(decoded: list[dict], tools: list[dict]) -> list[str]:
    """Our own schema screen (top-level only), used to split BFCL failures into schema_invalid vs
    wrong_call the same way for every category (BFCL's parallel checker reports both as
    cannot_find_match)."""
    by = {t["function"]["name"]: t["function"] for t in tools}
    out = []
    for call in decoded:
        for name, args in call.items():
            f = by.get(name)
            if f is None:
                out.append(f"unknown_function:{name}")
                continue
            props = f["parameters"].get("properties", {})
            for r in f["parameters"].get("required", []):
                if r not in args:
                    out.append(f"missing_required:{name}.{r}")
            for k, v in args.items():
                if k not in props:
                    out.append(f"unexpected_param:{name}.{k}")
                elif props[k].get("type") in _PY_OK and not _PY_OK[props[k]["type"]](v):
                    out.append(f"type:{name}.{k}")
    return out


def outcome_after_check(check: dict, decoded: list[dict], tools: list[dict]) -> str:
    if check["valid"]:
        return "correct"
    if (check.get("error_type") or "").startswith("ast_decoder"):
        return "malformed_structure"
    return "schema_invalid" if schema_violations(decoded, tools) else "wrong_call"


# --------------------------------------------------------------------------- ground-truth canonicalisation

def _canon_value(v):
    """BFCL nests acceptable-value lists inside dict values: {"size": ["large"], ...}. Pick one
    acceptable value per inner key ("" acceptable = optional, omitted), recursively; plain arrays stay."""
    if isinstance(v, dict):
        out = {}
        for k, vs in v.items():
            vs = vs if isinstance(vs, list) else [vs]
            if not vs or "" in vs:
                continue
            out[k] = _canon_value(vs[0])
        return out
    if isinstance(v, list):
        return [_canon_value(x) if isinstance(x, dict) else x for x in v]
    return v


def canonical(gt, functions: list[dict] | None = None):
    """One concrete call list from BFCL's acceptable-value ground truth, names with "." -> "_" (what
    the model sees). A top-level parameter is omitted when "" is acceptable and it is not required
    (BFCL GT sometimes lists "" first for a required parameter, e.g. simple_python_200)."""
    req = {}
    for f in functions or []:
        req[f["name"]] = set(f["parameters"].get("required", []))
    calls = []
    for call in gt:
        for name, params in call.items():
            args = {}
            for k, vals in params.items():
                vals = vals if isinstance(vals, list) else [vals]
                non_empty = [v for v in vals if v != ""]
                if not non_empty or ("" in vals and k not in req.get(name, set())):
                    continue
                args[k] = _canon_value(non_empty[0])
            calls.append({name.replace(".", "_"): args})
    return calls


def unrepresentable(gt, tools: list[dict], functions: list[dict]) -> list[str]:
    """Parameters whose canonical ground-truth value is a string while the offered schema type is
    array/object/integer/number/boolean: such a value cannot be emitted as a schema-typed Qwen3.8
    tool-call argument (e.g. parallel_multiple_21 passes x="data['sales']" for an array of floats)."""
    by = {t["function"]["name"]: t["function"] for t in tools}
    bad = []
    for call in canonical(gt, functions):
        for name, args in call.items():
            props = by.get(name, {}).get("parameters", {}).get("properties", {})
            for k, v in args.items():
                t = (props.get(k) or {}).get("type", "string")
                if t != "string" and isinstance(v, str):
                    bad.append(f"{name}.{k}:{t}={v!r}")
    return bad
