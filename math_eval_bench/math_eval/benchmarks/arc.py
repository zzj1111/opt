"""ARC-Challenge benchmark (Clark et al., 2018) — science reasoning.

Evaluates science question answering on the Challenge set of ARC
(AI2 Reasoning Challenge). Multiple-choice format with 4-5 options.
"""

import re
from typing import List, Optional

from datasets import load_dataset

from .base import Benchmark, BenchmarkItem
from ..answer import extract_boxed, strip_thinking


class ARCChallengeBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "arc_challenge"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        items = []
        for i, row in enumerate(ds):
            choices = row["choices"]
            labels = choices["label"]
            texts = choices["text"]
            options_str = "\n".join(
                f"{label}. {text}" for label, text in zip(labels, texts)
            )
            items.append(
                BenchmarkItem(
                    id=str(i),
                    question=row["question"],
                    gold_answer=row["answerKey"],
                    metadata={"options": options_str},
                )
            )
        if subset:
            items = items[:subset]
        return items

    def build_prompt(self, item: BenchmarkItem) -> List[dict]:
        options = item.metadata["options"]
        user_msg = (
            "Answer the following multiple-choice science question. "
            "Think step by step, then put your answer letter in \\boxed{}.\n\n"
            f"Question: {item.question}\n\n"
            f"{options}\n\n"
            "Answer:"
        )
        return [
            {"role": "user", "content": user_msg},
        ]

    def extract_answer(self, text: str) -> Optional[str]:
        text = strip_thinking(text)
        # Try \boxed{} first
        ans = extract_boxed(text)
        if ans is not None:
            ans = ans.strip().upper()
            if len(ans) == 1 and ans in "ABCDE12345":
                return self._normalize_label(ans)

        # Regex: "the answer is (X)" pattern
        m = re.search(
            r"(?:the answer is|answer is|answer:)\s*\(?([A-Ea-e1-5])\)?",
            text,
            re.IGNORECASE,
        )
        if m:
            return self._normalize_label(m.group(1).upper())

        # Last single letter A-E in the text
        letters = re.findall(r"\b([A-Ea-e])\b", text)
        if letters:
            return self._normalize_label(letters[-1].upper())

        return None

    def _normalize_label(self, label: str) -> str:
        """Normalize numeric labels (1-5) to letter labels (A-E)."""
        mapping = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        return mapping.get(label, label.upper())

    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        if pred is None:
            return False
        return self._normalize_label(pred.strip()) == self._normalize_label(
            gold.strip()
        )
