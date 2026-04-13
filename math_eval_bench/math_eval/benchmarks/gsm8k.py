"""GSM8K benchmark (Cobbe et al., 2021)."""

from typing import List, Optional

from datasets import load_dataset

from .base import Benchmark, BenchmarkItem
from ..answer import extract_answer_math, strip_thinking, extract_last_number

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math problems step by step. "
    "Put your final answer in \\boxed{}."
)


def _extract_gsm8k_gold(answer_text: str) -> str:
    """Extract numeric answer from GSM8K's '#### 42' format."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


class GSM8KBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "gsm8k"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = []
        for i, row in enumerate(ds):
            items.append(BenchmarkItem(
                id=str(i),
                question=row["question"],
                gold_answer=_extract_gsm8k_gold(row["answer"]),
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
        from ..answer import normalize_numeric
        np_ = normalize_numeric(pred)
        ng_ = normalize_numeric(gold)
        if np_ is not None and ng_ is not None:
            return abs(np_ - ng_) < 1e-2
        return pred.strip() == gold.strip()
