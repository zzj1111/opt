"""Abstract backend interface for model inference."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GenerationResult:
    """Single generation result from a backend."""
    text: str
    prompt: str
    finish_reason: Optional[str] = None


class Backend(ABC):
    @abstractmethod
    def generate(self, prompts: List[str]) -> List[GenerationResult]:
        ...

    @abstractmethod
    def generate_chat(self, messages_list: List[List[dict]]) -> List[GenerationResult]:
        ...

    def generate_chat_n(
        self, messages_list: List[List[dict]], n: int
    ) -> List[List[GenerationResult]]:
        """Generate n samples per prompt. Returns List[List[GenerationResult]].

        Default: call generate_chat n times. Backends may override for efficiency.
        """
        all_results: List[List[GenerationResult]] = [[] for _ in messages_list]
        for _ in range(n):
            results = self.generate_chat(messages_list)
            for i, r in enumerate(results):
                all_results[i].append(r)
        return all_results
