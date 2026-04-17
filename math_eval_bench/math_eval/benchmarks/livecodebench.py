"""LiveCodeBench benchmark (Jain et al., 2024) — contamination-free coding eval.

LiveCodeBench collects LeetCode-style problems published after training cutoffs
to avoid contamination. Each release (v1-v6) adds newer problems.

Dataset: livecodebench/code_generation_lite (release_v6 = 1055 problems total).
Supports two problem types:
  - stdin: read from stdin, write to stdout (competition-style)
  - functional: implement a class method (LeetCode-style)

Environment variables:
  LCB_VERSION: which release to use (v1..v6, default v6)
  LCB_DATE_FROM: only include problems on/after this date (YYYY-MM-DD)
  LCB_DATE_TO:   only include problems on/before this date (YYYY-MM-DD)

Defaults: release_v6, 2024-07-01 onwards (post-Qwen3 cutoff).
"""

import base64
import json
import os
import pickle
import re
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from typing import List, Optional

from huggingface_hub import hf_hub_download

from .base import Benchmark, BenchmarkItem
from ..answer import strip_thinking

_VERSION_FILES = {
    "v1": "test.jsonl",
    "v2": "test2.jsonl",
    "v3": "test3.jsonl",
    "v4": "test4.jsonl",
    "v5": "test5.jsonl",
    "v6": "test6.jsonl",
}


def _decode_private_tests(raw: str) -> list[dict]:
    """Private tests are zlib+pickle+base64 encoded (or plain JSON for older releases)."""
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))
    except Exception:
        return []


def _extract_code(text: str) -> str:
    text = strip_thinking(text)
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def _run_stdin(code: str, test_input: str, expected: str, timeout: int = 10) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            ["python3", path],
            input=test_input,
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return result.stdout.strip() == expected.strip()
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run_functional(code: str, test_input: str, expected: str, timeout: int = 10) -> bool:
    """Functional tests call `Solution().method(*args)` and compare to expected.

    LCB convention: test_input is a Python literal for the args (often `json.loads`-able),
    expected is also a literal. We build a harness that calls the method.
    """
    harness = f"""
import json, sys

{code}

def _compare(a, b):
    try:
        return a == b
    except Exception:
        return str(a) == str(b)

_inp_raw = {json.dumps(test_input)}
_exp_raw = {json.dumps(expected)}

try:
    _inp = json.loads(_inp_raw)
except Exception:
    _inp = _inp_raw
try:
    _exp = json.loads(_exp_raw)
except Exception:
    _exp = _exp_raw

# Find Solution class and call its single public method
_sol = Solution()
_methods = [m for m in dir(_sol) if not m.startswith('_') and callable(getattr(_sol, m))]
if not _methods:
    sys.exit(1)
_fn = getattr(_sol, _methods[0])

if isinstance(_inp, list):
    _out = _fn(*_inp)
elif isinstance(_inp, dict):
    _out = _fn(**_inp)
else:
    _out = _fn(_inp)

sys.exit(0 if _compare(_out, _exp) else 1)
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        result = subprocess.run(
            ["python3", path], capture_output=True, timeout=timeout
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run_tests(code: str, test_cases: list[dict], testtype: str) -> bool:
    """Return True iff code passes ALL test cases."""
    if not code or not test_cases:
        return False
    for tc in test_cases:
        if testtype == "stdin":
            if not _run_stdin(code, tc["input"], tc["output"]):
                return False
        else:  # functional
            if not _run_functional(code, tc["input"], tc["output"]):
                return False
    return True


class LiveCodeBenchBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "livecodebench"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        version = os.environ.get("LCB_VERSION", "v6")
        if version not in _VERSION_FILES:
            raise ValueError(f"Unknown LCB_VERSION={version}, expected one of {list(_VERSION_FILES)}")

        date_from = os.environ.get("LCB_DATE_FROM", "2024-07-01")
        date_to = os.environ.get("LCB_DATE_TO", "9999-12-31")

        path = hf_hub_download(
            "livecodebench/code_generation_lite",
            _VERSION_FILES[version],
            repo_type="dataset",
        )

        items = []
        with open(path) as fp:
            for line in fp:
                r = json.loads(line)
                date = r.get("contest_date", "")[:10]
                if date and not (date_from <= date <= date_to):
                    continue

                public = json.loads(r["public_test_cases"]) if r["public_test_cases"] else []
                private = _decode_private_tests(r.get("private_test_cases", ""))
                tests = public + private
                if not tests:
                    continue

                testtype = tests[0]["testtype"]

                items.append(
                    BenchmarkItem(
                        id=r["question_id"],
                        question=r["question_content"],
                        gold_answer=json.dumps({"tests": tests, "testtype": testtype}),
                        metadata={
                            "starter_code": r.get("starter_code", ""),
                            "difficulty": r.get("difficulty", ""),
                            "platform": r.get("platform", ""),
                            "contest_date": r.get("contest_date", ""),
                            "testtype": testtype,
                        },
                    )
                )

        if subset:
            items = items[:subset]
        return items

    def build_prompt(self, item: BenchmarkItem) -> List[dict]:
        starter = (item.metadata or {}).get("starter_code", "")
        testtype = (item.metadata or {}).get("testtype", "stdin")

        if testtype == "functional":
            instruction = (
                "Solve the following problem by completing the Python class below. "
                "Put your complete solution in a ```python``` code block.\n"
            )
            if starter:
                instruction += f"\n```python\n{starter}```\n"
        else:
            instruction = (
                "Solve the following competitive-programming problem in Python. "
                "Read from standard input and write to standard output. "
                "Put your complete solution in a ```python``` code block.\n"
            )

        user_msg = instruction + f"\nProblem:\n{item.question}"
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
        return _run_tests(pred, meta["tests"], meta["testtype"])
