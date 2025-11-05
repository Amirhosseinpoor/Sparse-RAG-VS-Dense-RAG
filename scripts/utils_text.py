import re
import nltk
from typing import Dict

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def fuse_title_text(doc: Dict) -> str:
    title = (doc.get("title") or "").strip()
    text  = (doc.get("text") or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text

def simple_tokenize(s: str):
    return TOKEN_RE.findall((s or "").lower())
