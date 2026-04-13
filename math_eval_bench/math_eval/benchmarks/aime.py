"""AIME 2024 / 2025 benchmarks."""

from typing import List, Optional

from datasets import load_dataset

from .base import Benchmark, BenchmarkItem
from ..answer import extract_answer_integer, is_integer_equiv

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves competition math problems step by step. "
    "The answer is always an integer between 000 and 999 inclusive. "
    "Put your final answer in \\boxed{}."
)


class AIME2024Benchmark(Benchmark):
    @property
    def name(self) -> str:
        return "aime2024"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        items = []
        for row in ds:
            items.append(BenchmarkItem(
                id=row["ID"],
                question=row["Problem"],
                gold_answer=str(row["Answer"]),
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
        return extract_answer_integer(text)

    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        return is_integer_equiv(pred, gold)


class AIME2025Benchmark(Benchmark):
    @property
    def name(self) -> str:
        return "aime2025"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        ds1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        ds2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
        items = []
        for i, row in enumerate(ds1):
            items.append(BenchmarkItem(
                id=f"2025-I-{i+1}",
                question=row["question"],
                gold_answer=str(row["answer"]),
            ))
        for i, row in enumerate(ds2):
            items.append(BenchmarkItem(
                id=f"2025-II-{i+1}",
                question=row["question"],
                gold_answer=str(row["answer"]),
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
        return extract_answer_integer(text)

    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        return is_integer_equiv(pred, gold)
