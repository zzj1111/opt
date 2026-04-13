"""GPQA Diamond benchmark (Rein et al., 2023)."""

from typing import List, Optional

from datasets import load_dataset

from .base import Benchmark, BenchmarkItem
from ..answer import extract_answer_math, is_equiv

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves graduate-level science problems step by step. "
    "Put your final answer in \\boxed{}."
)


class GPQADiamondBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "gpqa_diamond"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        ds = load_dataset("hendrydong/gpqa_diamond", split="test")
        items = []
        for i, row in enumerate(ds):
            items.append(BenchmarkItem(
                id=str(i),
                question=row["problem"],
                gold_answer=row["solution"],
                metadata={"domain": row.get("domain")},
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
        # Gold is in \boxed{} format from hendrydong dataset
        gold_extracted = extract_answer_math(gold)
        if gold_extracted is None:
            gold_extracted = gold
        return is_equiv(pred, gold_extracted)
