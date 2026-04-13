"""Abstract benchmark interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class BenchmarkItem:
    """A single evaluation item."""
    id: str
    question: str
    gold_answer: str
    metadata: Optional[Dict] = None


class Benchmark(ABC):
    """Abstract benchmark that defines dataset loading, prompting, and scoring."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def load(self, subset: Optional[int] = None) -> List[BenchmarkItem]:
        """Load benchmark items."""
        ...

    @abstractmethod
    def build_prompt(self, item: BenchmarkItem) -> List[dict]:
        """Build chat messages for evaluation."""
        ...

    @abstractmethod
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract model answer from generation output."""
        ...

    @abstractmethod
    def is_correct(self, pred: Optional[str], gold: str) -> bool:
        """Check if predicted answer matches gold."""
        ...
