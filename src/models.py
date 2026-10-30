"""Pydantic models for datasets, search results and answers."""
import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A character span [first, last) in a file.

    Attributes:
        file_path: Path of the file.
        first_character_index: Index of the first character.
        last_character_index: Index after the last character.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without a reference answer.

    Attributes:
        question_id: Unique ID, a random UUID by default.
        question: Text of the question.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and answer.

    Attributes:
        sources: Sources needed to answer.
        answer: Reference answer.
    """

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A list of questions.

    Attributes:
        rag_questions: Questions, answered or not.
    """

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Sources retrieved for one question.

    Attributes:
        question_id: ID of the question.
        question: Text of the question.
        retrieved_sources: Retrieved sources, best first.
    """

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results for one question with an answer.

    Attributes:
        answer: Generated answer.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a dataset.

    Attributes:
        search_results: Results, one per question.
        k: Number of sources per question, positive.
    """

    search_results: list[MinimalSearchResults]
    k: int = Field(gt=0)


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and answers for a dataset.

    Attributes:
        search_results: Results with answers, one per question.
        k: Number of sources per question, positive.
    """

    search_results: list[MinimalAnswer]
    k: int = Field(gt=0)


class Chunk(MinimalSource):
    """A span of a corpus file with its text.

    Attributes:
        text: Content of the file in the span.
    """

    text: str


class ChunkList(BaseModel):
    """Chunks of the corpus, saved by the index command.

    Attributes:
        chunks: Chunks, by chunk index.
    """

    chunks: list[Chunk]


class BM25Index(BaseModel):
    """An inverted index of chunks for BM25 search.

    Attributes:
        chunk_token_counts: Number of tokens in each chunk, by chunk index.
        postings: Chunk index and term frequency for each term.
    """

    chunk_token_counts: list[int]
    postings: dict[str, list[tuple[int, int]]]
