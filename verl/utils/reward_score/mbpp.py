"""MBPP reward function for veRL RL training.

Evaluates Python code by running assert-based test cases from the dataset.
Compatible with the MBPP parquet format where ground_truth contains:
  {"test_list": ["assert func(...) == ...", ...], "test_setup_code": "...", "solution": "..."}
"""

import json
import subprocess
import tempfile
import os


def compute_score(solution_str, ground_truth, timeout=10):
    """Score a code solution against MBPP test cases.

    Args:
        solution_str: Model-generated text (may contain ```python blocks).
        ground_truth: JSON string with test_list, test_setup_code, etc.
        timeout: Execution timeout in seconds.

    Returns:
        float: 1.0 if all tests pass, 0.0 otherwise.
    """
    # Extract code from markdown blocks if present
    code = _extract_code(solution_str)
    if not code.strip():
        return 0.0

    try:
        if isinstance(ground_truth, str):
            test_data = json.loads(ground_truth)
        else:
            test_data = ground_truth
    except (json.JSONDecodeError, TypeError):
        return 0.0

    test_list = test_data.get("test_list", [])
    test_setup_code = test_data.get("test_setup_code", "")

    if not test_list:
        return 0.0

    # Build the full test script
    parts = []
    if test_setup_code:
        parts.append(test_setup_code)
    parts.append(code.strip())
    parts.append("")  # blank line
    parts.extend(test_list)
    full_code = "\n".join(parts)

    return _run_code(full_code, timeout)


def _extract_code(text: str) -> str:
    """Extract Python code from model output."""
    # Remove thinking tags
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    if "```python" in text:
        return text.split("```python")[-1].split("```")[0]
    elif "```" in text:
        blocks = text.split("```")
        if len(blocks) >= 3:
            return blocks[1]
    return text


def _run_code(code: str, timeout: int) -> float:
    """Execute code in a subprocess and return 1.0 on success, 0.0 on failure."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
