"""MBPP reward: execute generated code against assert-based test cases."""

import json
import multiprocessing
import traceback


def _run_tests(code: str, test_list: list[str], timeout: int = 10) -> bool:
    """Execute code + assert statements in a subprocess with timeout."""
    # Combine generated code with all assert test cases
    full_code = code + "\n\n" + "\n".join(test_list)

    def _exec_fn(code_str):
        try:
            exec(code_str, {})
        except Exception:
            raise

    proc = multiprocessing.Process(target=_exec_fn, args=(full_code,))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return False

    return proc.exitcode == 0


def compute_score(completion: str, ground_truth, continuous=False):
    """Score a completion against MBPP test cases.

    Args:
        completion: Model-generated response (may contain markdown code blocks).
        ground_truth: JSON string of list of assert statements, e.g.
                      '["assert func(1)==2", "assert func(3)==4"]'
        continuous: If True, return per-test-case scores.

    Returns:
        (score, metadata)
    """
    # Extract python code from markdown block if present
    if "```python" in completion:
        code = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        code = completion.split("```")[1].split("```")[0]
    else:
        code = completion

    code = code.strip()

    try:
        if isinstance(ground_truth, str):
            parsed = json.loads(ground_truth)
        else:
            parsed = ground_truth

        # Support both formats: direct list or dict with "test_list" key
        if isinstance(parsed, dict):
            test_list = parsed.get("test_list", [])
            setup_code = parsed.get("test_setup_code", "")
            if setup_code:
                code = setup_code + "\n" + code
        elif isinstance(parsed, list):
            test_list = parsed
        else:
            test_list = []

        if not continuous:
            success = _run_tests(code, test_list, timeout=10)
            return float(success), {"passed": success}
        else:
            # Per-test scoring
            results = []
            for test in test_list:
                passed = _run_tests(code, [test], timeout=5)
                results.append(passed)

            score = sum(results) / max(len(results), 1)
            metadata = [{"test": t, "passed": p} for t, p in zip(test_list, results)]
            return score, metadata

    except Exception:
        traceback.print_exc(5)
        return 0.0, {"error": "execution_failed"}
