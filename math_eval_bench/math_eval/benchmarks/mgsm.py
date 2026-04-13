"""MGSM benchmark — multilingual math (10 languages), 250 per language."""

import json
import urllib.request
from typing import List, Optional

from .base import Benchmark, BenchmarkItem
from ..answer import strip_thinking, extract_boxed, extract_last_number, normalize_numeric

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math problems step by step. "
    "Put your final numeric answer in \\boxed{}."
)

MGSM_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]

# Raw MGSM data from the original repo
_BASE_URL = "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_{lang}.tsv"


def _load_mgsm_lang(lang: str) -> list[dict]:
    """Download and parse a single MGSM language TSV."""
    url = _BASE_URL.format(lang=lang)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return []
    items = []
    for line in text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            question = parts[0].strip()
            answer = parts[1].strip()
            items.append({"question": question, "answer": answer})
    return items


class MGSMBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "mgsm"

    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        items = []
        per_lang_limit = (subset // len(MGSM_LANGUAGES) + 1) if subset else None
        for lang in MGSM_LANGUAGES:
            rows = _load_mgsm_lang(lang)
            for i, row in enumerate(rows):
                if per_lang_limit and i >= per_lang_limit:
                    break
                items.append(BenchmarkItem(
                    id=f"{lang}_{i}",
                    question=row["question"],
                    gold_answer=row["answer"],
                    metadata={"language": lang},
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
        boxed = extract_boxed(text)
        if boxed is not None:
            return boxed.strip()
        return extract_last_number(text)

    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        if pred is None:
            return False
        np_ = normalize_numeric(pred)
        ng_ = normalize_numeric(gold)
        if np_ is not None and ng_ is not None:
            return abs(np_ - ng_) < 1e-2
        return pred.strip() == gold.strip()
