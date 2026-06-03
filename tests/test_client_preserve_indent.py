"""Tests for LLMClient.chat() preserve_indent parameter and the _strip_blank_edges helper.

The root-cause bug: LLMClient.chat() called .strip() on the API response text,
which removed leading spaces from the *first* line of a correctly-indented code
body.  For HumanEval, the model returns a body like:

    "    if x:\n        return 1\n    return 0"

After .strip() the first line loses its leading 4-space indent, making it
flush-left while every other line stays indented → IndentationError downstream.

The fix:
  - A helper ``_strip_blank_edges(s)`` that removes only leading/trailing blank
    lines and trailing whitespace per line, but preserves per-line leading
    indentation.
  - A ``preserve_indent`` keyword argument on ``chat()`` (default False) that
    switches from .strip() to _strip_blank_edges().
"""

import json
import pytest


# ---------------------------------------------------------------------------
# Unit tests for the _strip_blank_edges helper
# ---------------------------------------------------------------------------

from lib.evals.base import _strip_blank_edges


def test_preserve_indent_keeps_leading_spaces():
    """Helper preserves first-line indent and drops trailing blank line."""
    s = "    if x:\n        return 1\n    return 0\n\n"
    out = _strip_blank_edges(s)
    lines = out.split("\n")
    assert lines[0] == "    if x:", f"first line: {lines[0]!r}"
    assert lines[1] == "        return 1", f"second line: {lines[1]!r}"
    assert not out.endswith("\n\n"), "trailing blank lines must be removed"


def test_helper_strips_leading_blank_lines():
    """Helper removes blank lines at the top."""
    s = "\n\n    body_line\n"
    out = _strip_blank_edges(s)
    assert out == "    body_line"


def test_helper_strips_trailing_blank_lines():
    """Helper removes blank lines at the bottom."""
    s = "    first\n    second\n\n\n"
    out = _strip_blank_edges(s)
    assert not out.endswith("\n"), "should not end with newline after stripping trailing blanks"
    assert "    first" in out
    assert "    second" in out


def test_helper_strips_trailing_whitespace_per_line():
    """Helper strips trailing spaces/tabs from each line, not leading spaces."""
    s = "    line_one   \n    line_two\t\n"
    out = _strip_blank_edges(s)
    lines = out.split("\n")
    assert lines[0] == "    line_one", f"got: {lines[0]!r}"
    assert lines[1] == "    line_two", f"got: {lines[1]!r}"


def test_helper_preserves_internal_blank_lines():
    """Blank lines in the middle of the content are kept."""
    s = "    first\n\n    second\n"
    out = _strip_blank_edges(s)
    assert "\n\n" in out, "internal blank line must be preserved"


def test_helper_empty_string():
    """Empty input returns empty string."""
    assert _strip_blank_edges("") == ""


def test_helper_only_blank_lines():
    """Input that is all blank lines returns empty string."""
    assert _strip_blank_edges("\n\n\n") == ""


def test_helper_single_content_line_no_trailing_newline():
    """Single line without trailing newline is returned unchanged (just rstripped)."""
    s = "    return 42"
    assert _strip_blank_edges(s) == "    return 42"


# ---------------------------------------------------------------------------
# End-to-end: chat() with preserve_indent=True via monkeypatched HTTP
#
# We patch httpx.Client.post to return a canned response dict so we can test
# the full chat() code-path without a live server.
# ---------------------------------------------------------------------------

import httpx
from unittest.mock import MagicMock, patch

from lib.evals.base import LLMClient


def _make_mock_response(content: str) -> MagicMock:
    """Return a mock httpx.Response that yields a canned message content."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": content,
                    "reasoning_content": None,
                }
            }
        ]
    }
    return mock_resp


# Content that represents a correctly-indented code body returned by the model.
# The leading 4 spaces on the first line are the key invariant we are protecting.
_INDENTED_BODY = "    if threshold <= 0:\n        return False\n    return True\n\n"


def test_chat_preserve_indent_true_keeps_first_line_indent():
    """With preserve_indent=True, chat() must not strip leading spaces from content."""
    client = LLMClient("http://localhost:8080/v1", "test-model")
    mock_resp = _make_mock_response(_INDENTED_BODY)

    with patch.object(client._client, "post", return_value=mock_resp):
        result = client.chat(
            [{"role": "user", "content": "implement it"}],
            preserve_indent=True,
        )

    first_line = result.split("\n")[0]
    assert first_line == "    if threshold <= 0:", (
        f"first line should retain leading indent, got: {first_line!r}"
    )
    # Trailing blank lines must still be removed
    assert not result.endswith("\n"), f"trailing newline not stripped: {result!r}"


def test_chat_preserve_indent_false_strips_leading_spaces():
    """Default (preserve_indent=False) still strips leading whitespace from the response."""
    client = LLMClient("http://localhost:8080/v1", "test-model")
    mock_resp = _make_mock_response(_INDENTED_BODY)

    with patch.object(client._client, "post", return_value=mock_resp):
        result = client.chat(
            [{"role": "user", "content": "implement it"}],
        )

    # .strip() removes the leading spaces from the first line
    first_line = result.split("\n")[0]
    assert first_line == "if threshold <= 0:", (
        f"default path must strip leading whitespace, got: {first_line!r}"
    )


def test_chat_preserve_indent_reasoning_content_fallback():
    """preserve_indent=True also preserves indent in reasoning_content fallback path."""
    client = LLMClient("http://localhost:8080/v1", "test-model")

    # content is empty, reasoning_content carries the indented body
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": _INDENTED_BODY,
                }
            }
        ]
    }

    with patch.object(client._client, "post", return_value=mock_resp):
        result = client.chat(
            [{"role": "user", "content": "implement it"}],
            preserve_indent=True,
        )

    first_line = result.split("\n")[0]
    assert first_line == "    if threshold <= 0:", (
        f"reasoning_content fallback must also preserve indent, got: {first_line!r}"
    )


def test_chat_default_preserves_other_eval_behavior():
    """Non-code responses (like 'A' for MMLU) are unaffected by the default path."""
    client = LLMClient("http://localhost:8080/v1", "test-model")
    mock_resp = _make_mock_response("  A  ")

    with patch.object(client._client, "post", return_value=mock_resp):
        result = client.chat([{"role": "user", "content": "What is the answer?"}])

    assert result == "A", f"default .strip() must still work for non-code responses, got: {result!r}"
