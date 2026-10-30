from collections import Counter, defaultdict
from pathlib import Path

from src.chunker import chunk_python, chunk_text
from src.models import BM25Index, Chunk, ChunkList
from src.tokenizer import tokenize

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CHUNKERS = {
    ".py": chunk_python,
    ".md": chunk_text,
    ".txt": chunk_text,
}


def build_bm25_index(chunks: list[Chunk]) -> BM25Index:
    chunk_token_counts: list[int] = []
    postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)

    for chunk_id, chunk in enumerate(chunks):
        tokens = tokenize(chunk.text)
        chunk_token_counts.append(len(tokens))

        for token, terms_freq in Counter(tokens).items():
            postings[token].append((chunk_id, terms_freq))

    return BM25Index(chunk_token_counts=chunk_token_counts, postings=postings)


def build_index(max_chunk_size: int) -> None:
    chunks: list[Chunk] = []

    for path in sorted(RAW_DIR.rglob("*")):
        chunker = CHUNKERS.get(path.suffix)
        if chunker is None or not path.is_file():
            continue
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        chunks.extend(chunker(str(path), text, max_chunk_size))

    bm25_index = build_bm25_index(chunks)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_DIR / "chunks.json", "w", encoding="utf-8") as f:
        f.write(ChunkList(chunks=chunks).model_dump_json())

    with open(PROCESSED_DIR / "bm25_index.json", "w", encoding="utf-8") as f:
        f.write(bm25_index.model_dump_json())
