"""BIG-Bench Hard (BBH) benchmark — 3-shot CoT, 27 subtasks."""

import re
from typing import List, Optional

from datasets import load_dataset

from .base import Benchmark, BenchmarkItem
from ..answer import strip_thinking, extract_boxed

SYSTEM_PROMPT = (
    "You are a helpful assistant. Follow the examples and solve the task step by step. "
    "Put your final answer in \\boxed{}."
)

# All 27 BBH subtasks
BBH_SUBTASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]


class BBHBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "bbh"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        items = []
        for subtask in BBH_SUBTASKS:
            try:
                ds = load_dataset("lukaemon/bbh", subtask, split="test")
            except Exception:
                continue
            per_task_limit = (subset // len(BBH_SUBTASKS) + 1) if subset else None
            for i, row in enumerate(ds):
                if per_task_limit and i >= per_task_limit:
                    break
                items.append(BenchmarkItem(
                    id=f"{subtask}_{i}",
                    question=row["input"],
                    gold_answer=row["target"],
                    metadata={"subtask": subtask},
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
        # Try \boxed{...} with balanced brace matching (handles \boxed{\text{False}})
        boxed = extract_boxed(text)
        if boxed is not None:
            # Strip LaTeX wrappers like \text{...}, \textbf{...}, \mathrm{...}
            cleaned = re.sub(r"\\(?:text|textbf|mathrm)\{([^}]*)\}", r"\1", boxed)
            return cleaned.strip()
        # Try "the answer is ..."
        m = re.search(r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)", text, re.I)
        if m:
            return m.group(1).strip()
        # Last line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else None

    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        if pred is None:
            return False
        # Normalize: lowercase, strip parentheses
        p = pred.strip().lower().strip("(). ")
        g = gold.strip().lower().strip("(). ")
        return p == g
