"""MBPP benchmark (Austin et al., 2021) — execution-based.

Prompt format aligned with RL training reward function to ensure
consistent evaluation.
"""

import json
import os
import re
import subprocess
import tempfile
from typing import List, Optional

from datasets import load_dataset

from .base import Benchmark, BenchmarkItem
from ..answer import strip_thinking


def _extract_code(text: str) -> str:
    """Extract Python code from model output.

    Aligned with RL training reward function (verl/utils/reward_score/mbpp.py):
    takes the last ```python``` block.
    """
    text = strip_thinking(text)
    # Collect all python code blocks, take the last one (most likely the final answer)
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1]
    # Fallback: any code block
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1]
    return text


def _run_tests(code: str, test_list: list[str], timeout: int = 10) -> bool:
    """Run test assertions against extracted code."""
    full_code = code.strip() + "\n\n" + "\n".join(test_list)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        result = subprocess.run(['python3', tmp_path], capture_output=True, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class MBPPBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "mbpp"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        items = []
        for i, row in enumerate(ds):
            items.append(BenchmarkItem(
                id=str(row.get("task_id", i)),
                question=row["prompt"],
                gold_answer=json.dumps(row["test_list"]),
                metadata={"code": row.get("code", "")},
            ))
        if subset:
            items = items[:subset]
        return items

    def build_prompt(self, item: BenchmarkItem) -> List[dict]:
        """Build prompt aligned with RL training format.

        RL training uses:
          "Problem: {description}\n\nTest case:\nassert func(...) == ...\n\n
           Write ONLY the Python code (function definition)."
        """
        test_list = json.loads(item.gold_answer)
        first_test = test_list[0] if test_list else ""

        user_msg = (
            "You are a Python programmer. "
            "Write a function to solve the following problem.\n\n"
            f"Problem: {item.question}\n\n"
            f"Test case:\n{first_test}\n\n"
            "Write ONLY the Python code (function definition). "
            "Do not include any explanation."
        )
        return [
            {"role": "user", "content": user_msg},
        ]

    def extract_answer(self, text: str) -> Optional[str]:
        return _extract_code(text)

    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        if pred is None:
            return False
        try:
            test_list = json.loads(gold)
        except Exception:
            return False
        return _run_tests(pred, test_list)
