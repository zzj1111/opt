"""OlympiadBench (OE_MM_MATHS_EN_COMP) - text-only open-ended math competition problems."""

from typing import List, Optional

from datasets import load_dataset

from .base import Benchmark, BenchmarkItem
from ..answer import extract_answer_math, is_equiv

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves competition math problems step by step. "
    "Put your final answer in \\boxed{}."
)

MATH_FIELDS = {"Geometry", "Algebra", "Combinatorics", "Number Theory"}


class OlympiadBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "olympiadbench"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        ds = load_dataset("lmms-lab/OlympiadBench", split="test_en")
        items = []
        for row in ds:
            # Filter: math fields, open-ended (has answer), text-only (no images)
            if row["subfield"] not in MATH_FIELDS:
                continue
            if row["answer_type"] is None or row["final_answer"] is None:
                continue
            if row["images"] and len(row["images"]) > 0:
                continue

            gold = row["final_answer"]
            # final_answer is a list; join with comma for multi-answer
            if isinstance(gold, list):
                gold = ", ".join(str(a) for a in gold)

            context = row.get("context") or ""
            question = row["question"]
            if context:
                question = context + "\n\n" + question

            items.append(BenchmarkItem(
                id=str(row["question_id"]),
                question=question,
                gold_answer=gold,
                metadata={
                    "subfield": row["subfield"],
                    "answer_type": row["answer_type"],
                    "is_multiple_answer": row.get("is_multiple_answer"),
                    "unit": row.get("unit"),
                    "error": row.get("error"),
                },
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
        if pred is None:
            return False
        # Handle multi-answer: check each gold answer
        gold_parts = [g.strip() for g in gold.split(",")]
        if len(gold_parts) > 1:
            pred_parts = [p.strip() for p in pred.split(",")]
            if len(pred_parts) != len(gold_parts):
                # try as single equiv
                return is_equiv(pred, gold)
            # Try both original order and sorted order to handle (a,b) vs (b,a)
            if all(is_equiv(p, g) for p, g in zip(pred_parts, gold_parts)):
                return True
            # Try matching as sets: each gold part matched by some pred part
            used = [False] * len(pred_parts)
            for g in gold_parts:
                matched = False
                for j, p in enumerate(pred_parts):
                    if not used[j] and is_equiv(p, g):
                        used[j] = True
                        matched = True
                        break
                if not matched:
                    return False
            return True
        return is_equiv(pred, gold)
