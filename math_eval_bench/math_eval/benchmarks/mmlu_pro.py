"""MMLU-Pro benchmark (Wang et al., 2024) — 5-shot CoT MCQ."""

from typing import List, Optional

from datasets import load_dataset

from .base import Benchmark, BenchmarkItem
from ..answer import strip_thinking

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following multiple-choice question "
    "step by step. Put your final answer as a single letter (A-J) in \\boxed{}."
)


def _format_choices(options: list[str]) -> str:
    letters = "ABCDEFGHIJ"
    return "\n".join(f"({letters[i]}) {opt}" for i, opt in enumerate(options))


class MMLUProBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "mmlu_pro"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        items = []
        for i, row in enumerate(ds):
            letters = "ABCDEFGHIJ"
            gold_idx = row["answer_index"]
            gold_letter = letters[gold_idx] if isinstance(gold_idx, int) else str(row["answer"])
            items.append(BenchmarkItem(
                id=str(i),
                question=row["question"],
                gold_answer=gold_letter,
                metadata={
                    "options": row["options"],
                    "category": row.get("category"),
                },
            ))
        if subset:
            items = items[:subset]
        return items

    def build_prompt(self, item: BenchmarkItem) -> List[dict]:
        choices = _format_choices(item.metadata["options"])
        user_msg = f"{item.question}\n\n{choices}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def extract_answer(self, text: str) -> Optional[str]:
        import re
        text = strip_thinking(text)
        # Try \boxed{X}
        m = re.findall(r"\\boxed\{([A-Ja-j])\}", text)
        if m:
            return m[-1].upper()
        # Try "answer is (X)" or "answer is X"
        m = re.search(r"(?:answer is|answer:)\s*\(?([A-Ja-j])\)?", text, re.I)
        if m:
            return m.group(1).upper()
        # Last standalone letter
        m = re.findall(r"\b([A-J])\b", text)
        if m:
            return m[-1]
        return None

    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        if pred is None:
            return False
        return pred.strip().upper() == gold.strip().upper()
