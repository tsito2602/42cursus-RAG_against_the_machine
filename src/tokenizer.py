import re

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []

    for word in WORD_RE.findall(text):
        tokens.append(word.lower())
        parts = [p.lower() for p in PART_RE.findall(word)]

        if len(parts) > 1:
            tokens.extend(parts)

    return tokens
