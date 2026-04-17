"""HumanEval+ benchmark (EvalPlus, Liu et al., 2023) — execution-based code generation.

HumanEval+ extends OpenAI's HumanEval with ~80x more test cases to catch
solutions that pass the original weak tests but have subtle bugs.

Dataset: evalplus/humanevalplus (164 problems, same prompts as HumanEval).
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
    text = strip_thinking(text)
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def _run_check(code: str, test_code: str, entry_point: str, timeout: int = 30) -> bool:
    full_code = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
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


class HumanEvalPlusBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "humaneval_plus"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        ds = load_dataset("evalplus/humanevalplus", split="test")
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
                        }
                    ),
                    metadata={"entry_point": row["entry_point"]},
                )
            )
        if subset:
            items = items[:subset]
        return items

    def build_prompt(self, item: BenchmarkItem) -> List[dict]:
        user_msg = (
            "Complete the following Python function. "
            "Write ONLY the complete function implementation "
            "(including the function signature and docstring provided). "
            "Do not include any tests, examples, or explanation.\n\n"
            f"```python\n{item.question}```"
        )
        return [{"role": "user", "content": user_msg}]

    def extract_answer(self, text: str) -> Optional[str]:
        code = _extract_code(text)
        return code if code else None

    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        if pred is None:
            return False
        try:
            meta = json.loads(gold)
        except Exception:
            return False
        return _run_check(pred, meta["test"], meta["entry_point"])
