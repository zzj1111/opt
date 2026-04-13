"""C-Eval benchmark — Chinese knowledge MCQ, val split (has labels)."""

import re
from typing import List, Optional

from datasets import load_dataset, get_dataset_config_names

from .base import Benchmark, BenchmarkItem
from ..answer import strip_thinking

SYSTEM_PROMPT = (
    "你是一个知识渊博的助手。请逐步分析以下选择题，"
    "然后将最终答案以 \\boxed{} 的形式给出，例如 \\boxed{A}。"
)


def _format_choices(row: dict) -> str:
    parts = []
    for key in ["A", "B", "C", "D"]:
        val = row.get(key, "")
        if val:
            parts.append(f"({key}) {val}")
    return "\n".join(parts)


class CEvalBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "ceval"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        # Load all subjects
        configs = get_dataset_config_names("ceval/ceval-exam")
        items = []
        per_subject_limit = (subset // len(configs) + 1) if subset else None
        for subject in sorted(configs):
            try:
                ds = load_dataset("ceval/ceval-exam", subject, split="val")
            except Exception:
                continue
            for i, row in enumerate(ds):
                if per_subject_limit and i >= per_subject_limit:
                    break
                choices = _format_choices(row)
                items.append(BenchmarkItem(
                    id=f"{subject}_{i}",
                    question=f"{row['question']}\n\n{choices}",
                    gold_answer=row["answer"],
                    metadata={"subject": subject},
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
        text = strip_thinking(text)
        m = re.findall(r"\\boxed\{([A-Da-d])\}", text)
        if m:
            return m[-1].upper()
        m = re.search(r"(?:答案[是为：:]*|answer[:\s]*)\s*\(?([A-Da-d])\)?", text, re.I)
        if m:
            return m.group(1).upper()
        m = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text)
        if m:
            return m[-1]
        return None

    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        if pred is None:
            return False
        return pred.strip().upper() == gold.strip().upper()
