import math
import json
import ast
import re
from typing import List
import arabic_reshaper
from bidi.algorithm import get_display

def rtl(text: str) -> str:
    """Reshape & reorder Arabic for proper RTL display."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two equal‐length vectors."""
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def wrap_arabic(text: str, max_chars: int = 70) -> List[str]:
    """Naive word-wrap for Arabic lines."""
    words = text.split()
    lines, buf = [], []
    for w in words:
        if sum(len(x) for x in buf) + len(w) + len(buf) > max_chars:
            lines.append(" ".join(buf))
            buf = [w]
        else:
            buf.append(w)
    if buf:
        lines.append(" ".join(buf))
    return lines

# First occurrence (kept verbatim)
def _clean_user_question(raw: str) -> str:
    lowered = raw.lstrip().lower()
    if lowered.startswith(("سؤال:", "qa:")):
        return raw.split(":", 1)[1].lstrip()
    return raw

def strip_unsupported(text: str) -> str:
    """
    Remove any character that is not:
      - Arabic letters (U+0600–U+06FF)
      - Basic Latin letters/digits/punctuation (U+0000–U+007F)
      - Common Arabic punctuation: ، ؟ ! - (and space)
    This effectively strips emojis and other symbols that the Arabic font cannot render.
    """
    # Allow U+0600..U+06FF (Arabic), U+0000..U+007F (Basic Latin),
    # and the Arabic comma (U+060C) and question mark (U+061F) and exclamation (U+0021) and dash/hyphen.
    return re.sub(r"[^\u0000-\u007F\u0600-\u06FF\u060C\u061F\u0021\u002D\s]", "", text)

def _clean_json_block(text: str) -> str:
    cleaned = re.sub(r"```[a-zA-Z]*\n?", "", text).strip()
    return cleaned.strip("`").strip()

def parse_quiz_json(raw_text: str):
    cleaned = _clean_json_block(raw_text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        fixed = re.sub(r"'", '"', cleaned)
        return json.loads(fixed)
    except Exception:
        pass
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        return None

# Second occurrence (also kept verbatim; will override the first by Python rules)
def _clean_user_question(raw: str) -> str:
    lowered = raw.lstrip().lower()
    if lowered.startswith(("سؤال:", "qa:")):
        return raw.split(":", 1)[1].strip()
    return raw.strip()
