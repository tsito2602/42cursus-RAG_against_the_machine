"""Tokenizer for BM25 search over code and text."""
import re

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens.

    Each word of letters, digits and underscores becomes a token. If the
    word has several parts in snake_case or camelCase, each part is also
    added as a token, e.g. "load_lora" gives "load_lora", "load", "lora".

    Args:
        text: Text to tokenize.

    Returns:
        The tokens in order of appearance.
    """
    tokens: list[str] = []

    for word in WORD_RE.findall(text):
        tokens.append(word.lower())
        parts = [p.lower() for p in PART_RE.findall(word)]

        if len(parts) > 1:
            tokens.extend(parts)

    return tokens
