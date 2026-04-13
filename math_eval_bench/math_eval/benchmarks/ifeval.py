"""IFEval benchmark (Zhou et al., 2023) — instruction following."""

import json
from typing import List, Optional

from datasets import load_dataset

from .base import Benchmark, BenchmarkItem
from ..answer import strip_thinking
from .ifeval_checks import check_all_instructions


class IFEvalBenchmark(Benchmark):
    """IFEval: strict prompt-level instruction following evaluation.

    Each item has a set of verifiable constraints (e.g., "write at least 300 words",
    "include keyword X", "respond in all caps"). We check all constraints.
    """

    @property
    def name(self) -> str:
        return "ifeval"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        ds = load_dataset("google/IFEval", split="train")
        items = []
        for i, row in enumerate(ds):
            items.append(BenchmarkItem(
                id=str(i),
                question=row["prompt"],
                gold_answer=json.dumps({
                    "prompt": row["prompt"],
                    "instruction_id_list": row["instruction_id_list"],
                    "kwargs": row["kwargs"],
                }),
                metadata={"key": row.get("key")},
            ))
        if subset:
            items = items[:subset]
        return items

    def build_prompt(self, item: BenchmarkItem) -> List[dict]:
        return [{"role": "user", "content": item.question}]

    def extract_answer(self, text: str) -> Optional[str]:
        return strip_thinking(text)

    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        if pred is None:
            return False
        try:
            gold_data = json.loads(gold)
            return check_all_instructions(
                instruction_id_list=gold_data["instruction_id_list"],
                kwargs_list=gold_data["kwargs"],
                response=pred,
            )
        except Exception:
            return False
