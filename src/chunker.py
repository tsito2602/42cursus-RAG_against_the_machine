from src.models import MinimalSource


class Chunk(MinimalSource):
    """A span of a corpus file with its text.

    Attributes:
        text: Content of the file in the span.
    """

    text: str


def chunk_python(file_path: str, text: str, chunk_size: int) -> list[Chunk]:
    chunks: list[Chunk] = []

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


def chunk_text(file_path: str, text: str, chunk_size: int) -> list[Chunk]:
    chunks: list[Chunk] = []

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
