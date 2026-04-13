"""Answer extraction and comparison for math benchmarks.

Supports:
- LaTeX boxed answers (\boxed{...})
- Numeric answers (integers, floats)
- math_verify for symbolic equivalence
- prime_math grade_answer as final fallback
"""

import math
import re
from typing import Optional

# ---- extraction patterns ----
LAST_NUM_RE = re.compile(r"([-+]?\d[\d,]*(?:\.\d+)?)")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return THINK_RE.sub("", text).strip()


def extract_boxed(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} content using balanced brace counting.

    Handles arbitrary nesting depth (e.g. \\boxed{\\frac{\\sqrt{2}}{3}}).
    Aligned with prime_math's _last_boxed_only_string.
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                if left_brace_idx is not None:
                    return text[left_brace_idx + 1 : i].strip()
                return None
        i += 1

    return None


def extract_last_number(text: str) -> Optional[str]:
    """Extract the last number from text."""
    nums = LAST_NUM_RE.findall(text)
    if nums:
        return nums[-1].strip()
    return None


def normalize_numeric(s: str) -> Optional[float]:
    """Normalize a numeric string to float. Returns None on failure."""
    try:
        return float(s.replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def extract_answer_math(text: str) -> Optional[str]:
    """Extract answer for MATH-style problems.

    Priority: \\boxed{} from non-thinking part > \\boxed{} from full text > last number.
    """
    clean = strip_thinking(text)
    ans = extract_boxed(clean)
    if ans is not None:
        return ans
    # fallback: search full text (in case boxed is inside think block)
    ans = extract_boxed(text)
    if ans is not None:
        return ans
    return extract_last_number(clean)


def extract_answer_integer(text: str) -> Optional[str]:
    """Extract integer answer for AIME-style problems (0-999).

    Priority: \\boxed{} > last number.
    """
    text = strip_thinking(text)
    ans = extract_boxed(text)
    if ans is not None:
        num = normalize_numeric(ans)
        if num is not None and math.isfinite(num):
            return str(int(round(num)))
        return ans
    ans = extract_last_number(text)
    if ans is not None:
        num = normalize_numeric(ans)
        if num is not None and math.isfinite(num):
            return str(int(round(num)))
    return None


def extract_answer_numeric(text: str) -> Optional[str]:
    """Extract numeric answer for AMC-style problems.

    Priority: \\boxed{} > last number.
    """
    text = strip_thinking(text)
    ans = extract_boxed(text)
    if ans is not None:
        num = normalize_numeric(ans)
        if num is not None:
            return str(num)
        return ans
    return extract_last_number(text)


# ---- comparison functions ----

def _try_prime_math_grade(pred: str, gold: str) -> Optional[bool]:
    """Try using prime_math's grade_answer (sympy-based, same as RL training).
    Returns None if unavailable."""
    try:
        from verl.utils.reward_score.prime_math import grade_answer
        return grade_answer(pred, gold)
    except Exception:
        return None


def _try_math_verify(pred: str, gold: str) -> Optional[bool]:
    """Try using math_verify for symbolic comparison.
    Only trust True results; False may be a false negative."""
    try:
        from math_verify import verify, parse
        answer_parsed = parse(gold)
        pred_parsed = parse(pred)
        result = verify(answer_parsed, pred_parsed)
        if result is True:
            return True
        return None  # Don't trust False — fall through to other checks
    except Exception:
        return None


def _numeric_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    """Compare two strings as numbers."""
    na = normalize_numeric(a)
    nb = normalize_numeric(b)
    if na is not None and nb is not None:
        return abs(na - nb) < tol
    return False


def _normalize_latex(s: str) -> str:
    """Basic LaTeX normalization for string comparison."""
    s = s.strip()
    s = s.replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    s = s.replace("\\text{", "").replace("\\mathrm{", "").replace("\\textbf{", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\}$", "", s) if s.count("{") < s.count("}") else s
    return s


def is_equiv(pred: str, gold: str) -> bool:
    """Check if predicted answer is equivalent to gold answer.

    Tries (in order):
    1. Exact match
    2. prime_math grade_answer (sympy-based, aligned with RL training)
    3. math_verify symbolic comparison (only trust True)
    4. Numeric comparison
    5. Normalized string comparison
    """
    if pred is None or gold is None:
        return False

    pred = pred.strip()
    gold = gold.strip()

    # exact match
    if pred == gold:
        return True

    # prime_math grade_answer (same as RL training reward)
    result = _try_prime_math_grade(pred, gold)
    if result is True:
        return True

    # math_verify (only trust True)
    result = _try_math_verify(pred, gold)
    if result is True:
        return True

    # numeric
    if _numeric_equal(pred, gold):
        return True

    # normalized string
    if _normalize_latex(pred) == _normalize_latex(gold):
        return True

    return False


def is_integer_equiv(pred: str, gold: str) -> bool:
    """Check equivalence for integer answers (AIME)."""
    if pred is None or gold is None:
        return False
    np_ = normalize_numeric(pred)
    ng = normalize_numeric(gold)
    if np_ is not None and ng is not None:
        return int(round(np_)) == int(round(ng))
    return pred.strip() == gold.strip()


def is_numeric_equiv(pred: str, gold: str, tol: float = 1e-4) -> bool:
    """Check equivalence for numeric answers (AMC)."""
    if pred is None or gold is None:
        return False
    na = normalize_numeric(pred)
    nb = normalize_numeric(gold)
    if na is not None and nb is not None:
        return abs(na - nb) < tol
    return pred.strip() == gold.strip()
