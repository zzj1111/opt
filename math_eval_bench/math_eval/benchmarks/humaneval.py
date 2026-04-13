"""HumanEval benchmark (Chen et al., 2021) — execution-based code generation.

Evaluates functional correctness of code generated from docstrings.
Uses the canonical 164 problems from OpenAI's HumanEval dataset.
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


def _extract_code(text: str, entry_point: str) -> str:
    """Extract Python code from model output.

    Strategy: look for a function definition matching the entry point,
    then fall back to the last ```python``` block, then raw text.
    """
    text = strip_thinking(text)

    # Try to find ```python blocks, take the last one
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # Fallback: any code block
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # Fallback: raw text (might be just the function body)
    return text.strip()


def _run_check(code: str, test_code: str, timeout: int = 15) -> bool:
    """Run HumanEval test cases against generated code."""
    full_code = code + "\n\n" + test_code + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python3", tmp_path], capture_output=True, timeout=timeout
        )
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


class HumanEvalBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "humaneval"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        ds = load_dataset("openai/openai_humaneval", split="test")
        items = []
        for row in ds:
            items.append(
                BenchmarkItem(
                    id=row["task_id"],
                    question=row["prompt"],
                    gold_answer=json.dumps(
                        {
                            "test": row["test"],
                            "entry_point": row["entry_point"],
                            "canonical_solution": row.get("canonical_solution", ""),
                        }
                    ),
                    metadata={
                        "entry_point": row["entry_point"],
                        "prompt": row["prompt"],
                    },
                )
            )
        if subset:
            items = items[:subset]
        return items

    def build_prompt(self, item: BenchmarkItem) -> List[dict]:
        meta = json.loads(item.gold_answer)
        entry_point = meta["entry_point"]
        prompt = item.question

        user_msg = (
            "Complete the following Python function. "
            "Write ONLY the complete function implementation "
            "(including the function signature and docstring provided). "
            "Do not include any tests, examples, or explanation.\n\n"
            f"```python\n{prompt}```"
        )
        return [
            {"role": "user", "content": user_msg},
        ]

    def extract_answer(self, text: str) -> Optional[str]:
        # We need entry_point but extract_answer only gets text.
        # Just extract the code block; correctness check handles the rest.
        text = strip_thinking(text)
        blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
        return text.strip()

    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        if pred is None:
            return False
        try:
            meta = json.loads(gold)
        except Exception:
            return False

        test_code = meta["test"]
        entry_point = meta["entry_point"]

        # The test code calls check(entry_point), so we need the function defined.
        # If pred doesn't include the signature, prepend the prompt.
        code = pred

        return _run_check(code, test_code)
