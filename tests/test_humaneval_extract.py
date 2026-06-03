"""Tests for HumanEval code extraction and program assembly.

Covers the full-function path (reasoning/chat models like gpt-oss) and the
legacy body-only path, ensuring both produce a compilable executable program.
"""

import ast
import textwrap

import pytest

from lib.evals.humaneval import build_executable_program, _extract_completion


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Realistic HumanEval prompt for HumanEval/0 (has_close_elements)
PROMPT_HCE = textwrap.dedent("""\
    from typing import List


    def has_close_elements(numbers: List[float], threshold: float) -> bool:
        \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
        given threshold.
        >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
        False
        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
        True
        \"\"\"
""")

TEST_HCE = textwrap.dedent("""\
    def check(candidate):
        assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
        assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
        assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
        assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
        assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
        assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
        assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
""")

ENTRY_HCE = "has_close_elements"

# Realistic HumanEval prompt for HumanEval/1 (separate_paren_groups)
PROMPT_SEP = textwrap.dedent("""\
    from typing import List


    def separate_paren_groups(paren_string: str) -> List[str]:
        \"\"\" Input to this function is a string containing multiple groups of nested parentheses.
        Your goal is to separate those group into separate strings and return the list of those.
        Separate groups are balanced (each open brace is properly closed) and not nested within each other.
        Ignore any spaces in the input string.
        >>> separate_paren_groups('( ) (( )) (( )( ))')
        ['()', '(())', '(()())']
        \"\"\"
""")

TEST_SEP = textwrap.dedent("""\
    def check(candidate):
        assert candidate('(()()) ((())) () ((())(()))') == ['(()())', '((()))', '()', '((())(()))']
        assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
        assert candidate('(()(())((())))') == ['(()(())((())))']
        assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
""")

ENTRY_SEP = "separate_paren_groups"


# ---------------------------------------------------------------------------
# Case 1: Full function inside a ```python fence (gpt-oss style)
# ---------------------------------------------------------------------------

def test_full_function_in_fence_compiles_and_defines_entry_point():
    """Reasoning model emits a prose preamble + ```python fence with a full def."""
    raw_response = textwrap.dedent("""\
        Here is the solution:
        ```python
        def has_close_elements(numbers, threshold):
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    if abs(numbers[i] - numbers[j]) < threshold:
                        return True
            return False
        ```
    """)

    prog = build_executable_program(PROMPT_HCE, raw_response, TEST_HCE, ENTRY_HCE)

    # Must compile without errors
    compile(prog, "<test>", "exec")

    # Must define the entry point
    assert "def has_close_elements" in prog

    # Must include the test harness
    assert "def check" in prog
    assert f"check({ENTRY_HCE})" in prog


def test_full_function_in_fence_executes_correctly():
    """Assembled program for a correct full-function response actually passes."""
    raw_response = textwrap.dedent("""\
        Here is the solution:
        ```python
        def has_close_elements(numbers, threshold):
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    if abs(numbers[i] - numbers[j]) < threshold:
                        return True
            return False
        ```
    """)

    prog = build_executable_program(PROMPT_HCE, raw_response, TEST_HCE, ENTRY_HCE)
    # Execute in a clean namespace to verify correctness
    ns = {}
    exec(compile(prog, "<test>", "exec"), ns)  # must not raise


# ---------------------------------------------------------------------------
# Case 2: Body-only output (legacy path)
# ---------------------------------------------------------------------------

def test_body_only_unindented_compiles():
    """Legacy: model emits a body at column 0 (no outer indent), no def line.

    Real completion models typically return code at column 0.  The assembler
    must re-indent to 4 spaces before concatenating with the prompt stub.
    """
    raw_response = (
        "for i in range(len(numbers)):\n"
        "    for j in range(i + 1, len(numbers)):\n"
        "        if abs(numbers[i] - numbers[j]) < threshold:\n"
        "            return True\n"
        "return False\n"
    )

    prog = build_executable_program(PROMPT_HCE, raw_response, TEST_HCE, ENTRY_HCE)

    compile(prog, "<test>", "exec")
    assert "def has_close_elements" in prog
    assert "def check" in prog
    assert f"check({ENTRY_HCE})" in prog


def test_body_only_executes_correctly():
    """Legacy body-only assembled program runs without error."""
    raw_response = (
        "for i in range(len(numbers)):\n"
        "    for j in range(i + 1, len(numbers)):\n"
        "        if abs(numbers[i] - numbers[j]) < threshold:\n"
        "            return True\n"
        "return False\n"
    )

    prog = build_executable_program(PROMPT_HCE, raw_response, TEST_HCE, ENTRY_HCE)
    ns = {}
    exec(compile(prog, "<test>", "exec"), ns)


def test_body_only_single_return_compiles():
    """Body-only for a simple sort function."""
    prompt = textwrap.dedent("""\
        from typing import List


        def sort_numbers(numbers: List[float]) -> List[float]:
            \"\"\" Input is a space-delimited string of numberals from 'zero' to 'nine'.
            Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
            Return the string with numbers sorted from smallest to largest
            >>> sort_numbers('three one five')
            'one three five'
            \"\"\"
    """)
    test = textwrap.dedent("""\
        def check(candidate):
            assert candidate([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    """)
    raw_response = "    return sorted(numbers)\n"

    prog = build_executable_program(prompt, raw_response, test, "sort_numbers")
    compile(prog, "<test>", "exec")


# ---------------------------------------------------------------------------
# Case 3: Full function WITHOUT a fence (plain text, no backticks)
# ---------------------------------------------------------------------------

def test_full_function_no_fence_compiles():
    """Model returns a plain-text full function definition (no markdown fences)."""
    raw_response = textwrap.dedent("""\
        def has_close_elements(numbers, threshold):
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    if abs(numbers[i] - numbers[j]) < threshold:
                        return True
            return False
    """)

    prog = build_executable_program(PROMPT_HCE, raw_response, TEST_HCE, ENTRY_HCE)

    compile(prog, "<test>", "exec")
    assert "def has_close_elements" in prog
    assert "def check" in prog
    assert f"check({ENTRY_HCE})" in prog


def test_full_function_no_fence_executes():
    """Plain-text full function assembled program runs correctly."""
    raw_response = textwrap.dedent("""\
        def has_close_elements(numbers, threshold):
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    if abs(numbers[i] - numbers[j]) < threshold:
                        return True
            return False
    """)

    prog = build_executable_program(PROMPT_HCE, raw_response, TEST_HCE, ENTRY_HCE)
    ns = {}
    exec(compile(prog, "<test>", "exec"), ns)


# ---------------------------------------------------------------------------
# Case 4: Full function with thinking tags (reasoning model chain-of-thought)
# ---------------------------------------------------------------------------

def test_full_function_with_thinking_tags_compiles():
    """Model output includes <think>...</think> block before the actual solution."""
    raw_response = textwrap.dedent("""\
        <think>
        I need to check if any two numbers in the list are within the threshold of each other.
        I'll use a nested loop approach.
        </think>

        ```python
        def has_close_elements(numbers, threshold):
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    if abs(numbers[i] - numbers[j]) < threshold:
                        return True
            return False
        ```
    """)

    prog = build_executable_program(PROMPT_HCE, raw_response, TEST_HCE, ENTRY_HCE)
    compile(prog, "<test>", "exec")
    assert "def has_close_elements" in prog


# ---------------------------------------------------------------------------
# Case 5: Full function with trailing if __name__ == "__main__" (should be stripped)
# ---------------------------------------------------------------------------

def test_full_function_with_main_block_stripped():
    """Trailing if __name__ == '__main__': block is stripped from full-function output."""
    raw_response = textwrap.dedent("""\
        ```python
        def has_close_elements(numbers, threshold):
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    if abs(numbers[i] - numbers[j]) < threshold:
                        return True
            return False

        if __name__ == "__main__":
            print(has_close_elements([1.0, 2.0], 0.5))
        ```
    """)

    prog = build_executable_program(PROMPT_HCE, raw_response, TEST_HCE, ENTRY_HCE)
    compile(prog, "<test>", "exec")
    assert "def has_close_elements" in prog
    # The __main__ block should not cause problems; check it compiles and runs
    ns = {}
    exec(compile(prog, "<test>", "exec"), ns)


# ---------------------------------------------------------------------------
# Case 6: Full function not mangled by body re-indentation
# ---------------------------------------------------------------------------

def test_full_function_indentation_not_mangled():
    """The critical regression: full-function output must NOT get re-indented.

    The old code stripped the def line then re-indented the body (treating it
    as a completion to be appended to the prompt stub). Combined with the prompt
    stub also containing the def line this produced duplicate indentation leading
    to IndentationError.
    """
    raw_response = textwrap.dedent("""\
        ```python
        def has_close_elements(numbers, threshold):
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    if abs(numbers[i] - numbers[j]) < threshold:
                        return True
            return False
        ```
    """)

    prog = build_executable_program(PROMPT_HCE, raw_response, TEST_HCE, ENTRY_HCE)

    # Must parse as valid Python (no IndentationError)
    try:
        ast.parse(prog)
    except IndentationError as e:
        pytest.fail(f"IndentationError in assembled program: {e}\n\nProgram:\n{prog}")
    except SyntaxError as e:
        pytest.fail(f"SyntaxError in assembled program: {e}\n\nProgram:\n{prog}")


# ---------------------------------------------------------------------------
# Case 7: Separate_paren_groups – a more complex body-only test
# ---------------------------------------------------------------------------

def test_separate_paren_groups_body_only_compiles():
    """Body-only for a multi-statement function body (unindented, column-0 style).

    The assembler must re-indent the entire body to 4 spaces.
    """
    raw_response = textwrap.dedent("""\
        result = []
        current = []
        depth = 0
        for ch in paren_string:
            if ch == ' ':
                continue
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
                if depth == 0:
                    result.append(''.join(current))
                    current = []
        return result
    """)

    prog = build_executable_program(PROMPT_SEP, raw_response, TEST_SEP, ENTRY_SEP)
    compile(prog, "<test>", "exec")
    assert "def separate_paren_groups" in prog


# ---------------------------------------------------------------------------
# _extract_completion unit tests (existing helper, not removed)
# ---------------------------------------------------------------------------

def test_extract_completion_strips_fence():
    """_extract_completion pulls code out of a markdown fence."""
    raw = "```python\n    return 42\n```"
    result = _extract_completion(raw, "foo")
    assert "return 42" in result


def test_extract_completion_strips_think_tags():
    """_extract_completion discards content before </think>."""
    raw = "<think>ignore this</think>\n    return 42"
    result = _extract_completion(raw, "foo")
    assert "return 42" in result
    assert "ignore this" not in result


# ---------------------------------------------------------------------------
# Case 8: Real gpt-oss clean indented body (the actual bug regression test)
# ---------------------------------------------------------------------------

def test_real_gptoss_indented_body_assembles_clean():
    """Real gpt-oss output: already-indented body must NOT have its indentation destroyed.

    gpt-oss returns a clean body (no def line, no fence) where every line is
    already indented correctly: top-level body at 4 spaces, nested at 8.
    The old code stripped the leading whitespace from the first line via
    response.strip(), then dedent + re-indent produced broken indentation.
    """
    prompt = ('from typing import List\n\n\n'
              'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n'
              '    """ Check if any two numbers are closer than threshold. """\n')
    body = ('    if threshold <= 0 or len(numbers) < 2:\n'
            '        return False\n'
            '    sorted_nums = sorted(numbers)\n'
            '    for i in range(len(sorted_nums) - 1):\n'
            '        if abs(sorted_nums[i + 1] - sorted_nums[i]) < threshold:\n'
            '            return True\n'
            '    return False')
    test_code = ('def check(candidate):\n'
                 '    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n'
                 '    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n')

    prog = build_executable_program(prompt, body, test_code, "has_close_elements")

    # Must compile without IndentationError
    compile(prog, "<t>", "exec")

    # Must run correctly (check() assertions pass)
    exec(prog, {})  # noqa: S102
