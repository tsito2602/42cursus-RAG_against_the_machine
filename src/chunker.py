from typing import TypeAlias
from src.models import Chunk

Chunks: TypeAlias = list[Chunk]


def chunk_python(file_path: str, text: str, chunk_size: int) -> Chunks:
    chunks: Chunks = []

    text_length = len(text)
    for first in range(0, text_length, chunk_size):
        last = min(first + chunk_size, text_length)
        piece = text[first:last]
        chunks.append(
            Chunk(
                file_path=file_path,
                first_character_index=first,
                last_character_index=last,
                text=piece,
            )
        )

    return chunks


def chunk_text(file_path: str, text: str, chunk_size: int) -> Chunks:
    chunks: Chunks = []

    text_length = len(text)
    for first in range(0, text_length, chunk_size):
        last = min(first + chunk_size, text_length)
        piece = text[first:last]
        chunks.append(
            Chunk(
                file_path=file_path,
                first_character_index=first,
                last_character_index=last,
                text=piece,
            )
        )

    return chunks
