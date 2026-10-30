"""Build the chunk and BM25 indexes of the corpus."""
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
    """Build a BM25 inverted index from chunks.

    Args:
        chunks: Chunks to index, by chunk index.

    Returns:
        The token count of each chunk and the postings of each token.
    """
    chunk_token_counts: list[int] = []
    postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)

    for chunk_id, chunk in enumerate(chunks):
        tokens = tokenize(chunk.text)
        chunk_token_counts.append(len(tokens))

        for token, term_freq in Counter(tokens).items():
            postings[token].append((chunk_id, term_freq))

    return BM25Index(chunk_token_counts=chunk_token_counts, postings=postings)


def build_index(max_chunk_size: int) -> None:
    """Chunk the files in RAW_DIR and save the indexes to PROCESSED_DIR.

    Only files with a suffix in CHUNKERS are indexed. The chunks are saved
    to chunks.json and the BM25 index to bm25_index.json.

    Args:
        max_chunk_size: Maximum number of characters per chunk.
    """
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {RAW_DIR}")

    chunks: list[Chunk] = []

    for path in sorted(RAW_DIR.rglob("*")):
        chunker = CHUNKERS.get(path.suffix)

        if chunker is None or not path.is_file():
            continue

        try:
            with open(path, encoding="utf-8", newline="") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError) as e:
            print(f"Warning: skipped {path}: {e}")
            continue

        chunks.extend(chunker(str(path), text, max_chunk_size))

    if not chunks:
        raise ValueError(f"No files to index were found in {RAW_DIR}")

    bm25_index = build_bm25_index(chunks)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_DIR / "chunks.json", "w", encoding="utf-8") as f:
        f.write(ChunkList(chunks=chunks).model_dump_json())

    with open(PROCESSED_DIR / "bm25_index.json", "w", encoding="utf-8") as f:
        f.write(bm25_index.model_dump_json())
