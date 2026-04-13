"""MATH-500 benchmark (Hendrycks et al., 2021)."""

from typing import List, Optional

from datasets import load_dataset

from .base import Benchmark, BenchmarkItem
from ..answer import extract_answer_math, is_equiv

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math problems step by step. "
    "Put your final answer in \\boxed{}."
)


class MATH500Benchmark(Benchmark):
    @property
    def name(self) -> str:
        return "math500"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = []
        for i, row in enumerate(ds):
            items.append(BenchmarkItem(
                id=row.get("unique_id", str(i)),
                question=row["problem"],
                gold_answer=row["answer"],
                metadata={"subject": row.get("subject"), "level": row.get("level")},
            ))
        if subset:
            items = items[:subset]
        return items

    def build_prompt(self, item: BenchmarkItem) -> List[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.question},
        ]

    def extract_answer(self, text: str) -> Optional[str]:
        return extract_answer_math(text)

    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        return is_equiv(pred, gold)
