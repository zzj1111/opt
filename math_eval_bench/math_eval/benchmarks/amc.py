"""AMC benchmark (AMC 10/12, 2022-2023)."""

from typing import List, Optional

from datasets import load_dataset

from .base import Benchmark, BenchmarkItem
from ..answer import extract_answer_numeric, is_numeric_equiv

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves AMC competition math problems step by step. "
    "Put your final answer in \\boxed{}."
)


class AMCBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "amc"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
        items = []
        for row in ds:
            items.append(BenchmarkItem(
                id=str(row["id"]),
                question=row["problem"],
                gold_answer=str(row["answer"]),
                metadata={"url": row.get("url")},
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
        return extract_answer_numeric(text)

    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        return is_numeric_equiv(pred, gold)
