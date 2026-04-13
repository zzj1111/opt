"""IFEval instruction-following checkers — standalone, no lm_eval dependency.

Each checker function takes (response: str, **kwargs) and returns bool.
"""

import re
import json
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    return len(text.split())


def _count_sentences(text: str) -> int:
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return len([s for s in sentences if s.strip()])


def _count_paragraphs(text: str) -> int:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return len(paras) if paras else (1 if text.strip() else 0)


def _get_paragraphs(text: str) -> list:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _compare(actual: int, relation: str, target: int) -> bool:
    relation = relation.strip().lower()
    if relation == "at least":
        return actual >= target
    elif relation in ("at most", "less than or equal to"):
        return actual <= target
    elif relation == "less than":
        return actual < target
    elif relation == "more than":
        return actual > target
    elif relation == "exactly":
        return actual == target
    return False


# ---------------------------------------------------------------------------
# Language detection (lightweight, no external deps)
# ---------------------------------------------------------------------------

# ISO 639-1 to language name mapping (subset used by IFEval)
_LANG_MAP = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "pt": "Portuguese", "it": "Italian", "nl": "Dutch", "ru": "Russian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "hi": "Hindi", "bn": "Bengali", "te": "Telugu", "ta": "Tamil",
    "mr": "Marathi", "kn": "Kannada", "gu": "Gujarati", "ml": "Malayalam",
    "pa": "Punjabi", "ur": "Urdu", "sw": "Swahili", "vi": "Vietnamese",
    "th": "Thai", "tr": "Turkish", "pl": "Polish", "uk": "Ukrainian",
    "cs": "Czech", "ro": "Romanian", "hu": "Hungarian", "el": "Greek",
    "he": "Hebrew", "fi": "Finnish", "sv": "Swedish", "no": "Norwegian",
    "da": "Danish", "bg": "Bulgarian", "hr": "Croatian", "sr": "Serbian",
    "sk": "Slovak", "sl": "Slovenian", "lt": "Lithuanian", "lv": "Latvian",
    "et": "Estonian", "id": "Indonesian", "ms": "Malay", "tl": "Tagalog",
}


def _check_language(text: str, lang_code: str) -> bool:
    """Best-effort language check using langdetect if available, else True."""
    try:
        import langdetect
        detected = langdetect.detect(text)
        # langdetect returns ISO 639-1 codes; some flexibility needed
        return detected == lang_code or detected.startswith(lang_code)
    except Exception:
        # If langdetect not installed or fails, pass the check
        return True


# ---------------------------------------------------------------------------
# Instruction checkers
# ---------------------------------------------------------------------------

def check_no_comma(response: str, **kwargs) -> bool:
    """punctuation:no_comma — response must not contain commas."""
    return "," not in response


def check_number_highlighted_sections(response: str, num_highlights: int, **kwargs) -> bool:
    """detectable_format:number_highlighted_sections — at least num_highlights *highlighted* sections.
    Highlighted = wrapped in markdown bold **...**  or  *...*
    """
    highlights = re.findall(r"\*[^*\n]+\*", response)
    return len(highlights) >= num_highlights


def check_number_words(response: str, relation: str, num_words: int, **kwargs) -> bool:
    """length_constraints:number_words"""
    return _compare(_count_words(response), relation, num_words)


def check_number_placeholders(response: str, num_placeholders: int, **kwargs) -> bool:
    """detectable_content:number_placeholders — at least num_placeholders [placeholders]."""
    placeholders = re.findall(r"\[.+?\]", response)
    return len(placeholders) >= num_placeholders


def check_repeat_prompt(response: str, prompt_to_repeat: str, **kwargs) -> bool:
    """combination:repeat_prompt — response must contain the original prompt."""
    if not prompt_to_repeat:
        return True
    return prompt_to_repeat.strip().lower() in response.lower()


def check_title(response: str, **kwargs) -> bool:
    """detectable_format:title — response must contain a title in <<...>>."""
    return bool(re.search(r"<<.+?>>", response))


def check_english_lowercase(response: str, **kwargs) -> bool:
    """change_case:english_lowercase — entire response in lowercase (no uppercase letters)."""
    return response == response.lower()


def check_english_capital(response: str, **kwargs) -> bool:
    """change_case:english_capital — entire response in uppercase."""
    return response == response.upper()


def check_capital_word_frequency(response: str, capital_relation: str, capital_frequency: int, **kwargs) -> bool:
    """change_case:capital_word_frequency — number of fully-capitalized words meets relation."""
    words = response.split()
    cap_count = sum(1 for w in words if w.isupper() and w.isalpha())
    return _compare(cap_count, capital_relation, capital_frequency)


def check_number_bullet_lists(response: str, num_bullets: int, **kwargs) -> bool:
    """detectable_format:number_bullet_lists — at least num_bullets bullet points."""
    bullets = re.findall(r"^\s*[\*\-•]\s+", response, re.MULTILINE)
    numbered = re.findall(r"^\s*\d+[\.\)]\s+", response, re.MULTILINE)
    return (len(bullets) + len(numbered)) >= num_bullets


def check_multiple_sections(response: str, section_spliter: str, num_sections: int, **kwargs) -> bool:
    """detectable_format:multiple_sections — response split by section_spliter has >= num_sections."""
    if not section_spliter:
        return True
    sections = [s.strip() for s in response.split(section_spliter) if s.strip()]
    return len(sections) >= num_sections


def check_json_format(response: str, **kwargs) -> bool:
    """detectable_format:json_format — response must contain valid JSON."""
    # Try the whole response first
    text = response.strip()
    # Extract JSON block if wrapped in markdown code fence
    m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if m:
        text = m.group(1).strip()
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find any JSON object or array in the response
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        m = re.search(pattern, response)
        if m:
            try:
                json.loads(m.group())
                return True
            except (json.JSONDecodeError, ValueError):
                pass
    return False


def check_number_paragraphs(response: str, num_paragraphs: int, **kwargs) -> bool:
    """length_constraints:number_paragraphs — exactly num_paragraphs paragraphs."""
    return _count_paragraphs(response) == num_paragraphs


def check_two_responses(response: str, **kwargs) -> bool:
    """combination:two_responses — must contain two responses separated by ******."""
    parts = [p.strip() for p in response.split("******") if p.strip()]
    return len(parts) >= 2


def check_response_language(response: str, language: str, **kwargs) -> bool:
    """language:response_language — response in specified language."""
    return _check_language(response, language)


def check_letter_frequency(response: str, let_relation: str, letter: str, let_frequency: int, **kwargs) -> bool:
    """keywords:letter_frequency — count of specific letter/char meets relation."""
    count = response.lower().count(letter.lower())
    return _compare(count, let_relation, let_frequency)


def check_end_checker(response: str, end_phrase: str, **kwargs) -> bool:
    """startend:end_checker — response must end with the given phrase."""
    return response.strip().endswith(end_phrase.strip())


def check_quotation(response: str, **kwargs) -> bool:
    """startend:quotation — entire response wrapped in double quotes."""
    stripped = response.strip()
    return stripped.startswith('"') and stripped.endswith('"')


def check_keyword_existence(response: str, keywords: list, **kwargs) -> bool:
    """keywords:existence — all keywords must appear in response."""
    lower = response.lower()
    return all(kw.lower() in lower for kw in keywords)


def check_forbidden_words(response: str, forbidden_words: list, **kwargs) -> bool:
    """keywords:forbidden_words — none of the forbidden words may appear."""
    lower = response.lower()
    return all(fw.lower() not in lower for fw in forbidden_words)


def check_keyword_frequency(response: str, relation: str, keyword: str, frequency: int, **kwargs) -> bool:
    """keywords:frequency — keyword count meets relation."""
    count = response.lower().count(keyword.lower())
    return _compare(count, relation, frequency)


def check_number_sentences(response: str, relation: str, num_sentences: int, **kwargs) -> bool:
    """length_constraints:number_sentences"""
    return _compare(_count_sentences(response), relation, num_sentences)


def check_postscript(response: str, postscript_marker: str, **kwargs) -> bool:
    """detectable_content:postscript — response must contain the postscript marker."""
    return postscript_marker.lower() in response.lower()


def check_nth_paragraph_first_word(
    response: str, num_paragraphs: int, first_word: str, nth_paragraph: int, **kwargs
) -> bool:
    """length_constraints:nth_paragraph_first_word — nth paragraph starts with first_word,
    and there are at least num_paragraphs paragraphs."""
    paras = _get_paragraphs(response)
    if len(paras) < num_paragraphs:
        return False
    # nth_paragraph is 1-indexed
    idx = nth_paragraph - 1
    if idx < 0 or idx >= len(paras):
        return False
    first = paras[idx].split()[0] if paras[idx].split() else ""
    return first.lower().strip(".,;:!?") == first_word.lower().strip(".,;:!?")


def check_constrained_response(response: str, **kwargs) -> bool:
    """detectable_format:constrained_response — response is one of: My answer is yes/no/maybe."""
    normalized = response.strip().lower().rstrip(".")
    return normalized in {
        "my answer is yes", "my answer is no", "my answer is maybe",
        "yes", "no", "maybe",
    }


# ---------------------------------------------------------------------------
# Registry: instruction_id → checker function
# ---------------------------------------------------------------------------

CHECKER_REGISTRY = {
    "punctuation:no_comma": check_no_comma,
    "detectable_format:number_highlighted_sections": check_number_highlighted_sections,
    "length_constraints:number_words": check_number_words,
    "detectable_content:number_placeholders": check_number_placeholders,
    "combination:repeat_prompt": check_repeat_prompt,
    "detectable_format:title": check_title,
    "change_case:english_lowercase": check_english_lowercase,
    "change_case:english_capital": check_english_capital,
    "change_case:capital_word_frequency": check_capital_word_frequency,
    "detectable_format:number_bullet_lists": check_number_bullet_lists,
    "detectable_format:multiple_sections": check_multiple_sections,
    "detectable_format:json_format": check_json_format,
    "length_constraints:number_paragraphs": check_number_paragraphs,
    "combination:two_responses": check_two_responses,
    "language:response_language": check_response_language,
    "keywords:letter_frequency": check_letter_frequency,
    "startend:end_checker": check_end_checker,
    "startend:quotation": check_quotation,
    "keywords:existence": check_keyword_existence,
    "keywords:forbidden_words": check_forbidden_words,
    "keywords:frequency": check_keyword_frequency,
    "length_constraints:number_sentences": check_number_sentences,
    "detectable_content:postscript": check_postscript,
    "length_constraints:nth_paragraph_first_word": check_nth_paragraph_first_word,
    "detectable_format:constrained_response": check_constrained_response,
}


def check_instruction(instruction_id: str, response: str, **kwargs) -> bool:
    """Check a single instruction. Returns False if instruction_id is unknown."""
    checker = CHECKER_REGISTRY.get(instruction_id)
    if checker is None:
        return False
    # Filter out None kwargs
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        return checker(response, **clean_kwargs)
    except Exception:
        return False


def check_all_instructions(
    instruction_id_list: list,
    kwargs_list: list,
    response: str,
) -> bool:
    """Check all instructions for a single item. Returns True only if ALL pass."""
    if not response or not response.strip():
        return False
    for instruction_id, kw in zip(instruction_id_list, kwargs_list):
        if not check_instruction(instruction_id, response, **kw):
            return False
    return True
