"""Benchmark registry."""

from .base import Benchmark
from .math500 import MATH500Benchmark
from .aime import AIME2024Benchmark, AIME2025Benchmark
from .amc import AMCBenchmark
from .olympiad import OlympiadBenchmark
from .gpqa import GPQADiamondBenchmark
from .gsm8k import GSM8KBenchmark
from .mbpp import MBPPBenchmark
from .ifeval import IFEvalBenchmark
from .mmlu_pro import MMLUProBenchmark
from .bbh import BBHBenchmark
from .mgsm import MGSMBenchmark
from .ceval import CEvalBenchmark
from .humaneval import HumanEvalBenchmark
from .arc import ARCChallengeBenchmark

BENCHMARKS = {
    "math500": MATH500Benchmark,
    "gsm8k": GSM8KBenchmark,
    "mbpp": MBPPBenchmark,
    "humaneval": HumanEvalBenchmark,
    "ifeval": IFEvalBenchmark,
    "mmlu_pro": MMLUProBenchmark,
    "bbh": BBHBenchmark,
    "mgsm": MGSMBenchmark,
    "ceval": CEvalBenchmark,
    "amc": AMCBenchmark,
    "aime2024": AIME2024Benchmark,
    "aime2025": AIME2025Benchmark,
    "olympiadbench": OlympiadBenchmark,
    "gpqa_diamond": GPQADiamondBenchmark,
    "arc_challenge": ARCChallengeBenchmark,
}


def get_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: {name}. "
            f"Available: {list(BENCHMARKS.keys())}"
        )
    return BENCHMARKS[name]()
