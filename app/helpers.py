
import json, re
import ast
from typing import List
import math
import ast
from typing import List
import arabic_reshaper
from bidi.algorithm import get_display
import google.generativeai as genai
import os

def _clean_user_question(raw: str) -> str:
    l = raw.strip().lower()
    return raw.split(':',1)[1].strip() if l.startswith(('سؤال:','qa:')) else raw.strip()

def _clean_json_block(text: str) -> str:
    import re
    cleaned = re.sub(r"```[a-zA-Z]*\n?", "", text).strip()
    return cleaned.strip("`").strip()

def parse_quiz_json(raw_text: str):
    cleaned = _clean_json_block(raw_text).replace("'", '"')
    try:
        return json.loads(cleaned)
    except Exception:
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            return None 
def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    n1  = math.sqrt(sum(x*x for x in a))
    n2  = math.sqrt(sum(y*y for y in b))
    return dot/(n1*n2) if n1 and n2 else 0.0
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
def rtl(text: str) -> str:
    """Reshape & reorder Arabic for proper RTL display."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

def configure_gemini():
    genai.configure(os.getenv("GEMINI_API_KEY"))

def embed(text: str):
    res = genai.embed_content(model="text-embedding-004", content=text)
    emb = res.get("embedding")
    return emb["values"] if isinstance(emb, dict) and "values" in emb else emb
