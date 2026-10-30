from fire import Fire
from pydantic import ValidationError

from src.evaluator import (
    MAX_SOURCE_LENGTH,
    find_too_long_sources,
    recall_at_k,
)
from src.indexer import PROCESSED_DIR, build_index
from src.models import AnsweredQuestion, RagDataset, StudentSearchResults

DEFAULT_K = (1, 3, 5, 10)


class RagCLI:
    """Command line interface of the RAG system, run with Python Fire."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Chunk the files in data/raw/ and save the index to data/processed/.

        Args:
            max_chunk_size: Maximum number of characters per chunk.
        """
        try:
            build_index(max_chunk_size)
        except (OSError, ValueError) as e:
            print(f"Error: {e}")
            return

        print(f"Ingestion complete! Indices saved under {PROCESSED_DIR}/")

    def search(self, query: str, k: int) -> None:
        """Print the top-k sources for a single query.

        Args:
            query: Question to search for.
            k: Number of sources to retrieve.
        """

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        """Search every question in a dataset and save StudentSearchResults.

        Args:
            dataset_path: Path to the dataset JSON.
            k: Number of sources to retrieve per question.
            save_directory: Directory to write the search results JSON to.
        """

    def answer(self, query: str, k: int) -> None:
        """Print an answer to a single query from the retrieved context.

        Args:
            query: Question to answer.
            k: Number of sources to use as context.
        """

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Answer every question in search results and save the answers.

        The output is a StudentSearchResultsAndAnswer JSON.

        Args:
            student_search_results_path: Path to the search results JSON.
            save_directory: Directory to write the answers JSON to.
        """

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Print recall@k of the search results, as moulinette does.

        Args:
            student_search_results_path: Path to the search results JSON.
            dataset_path: Path to the answered dataset JSON.
        """
        try:
            with open(student_search_results_path, encoding="utf-8") as f:
                student = StudentSearchResults.model_validate_json(f.read())
            with open(dataset_path, encoding="utf-8") as f:
                dataset = RagDataset.model_validate_json(f.read())
        except (OSError, ValidationError) as e:
            print(f"Error: {e}")
            return

        gold_questions = [
            q for q in dataset.rag_questions if isinstance(q, AnsweredQuestion)
        ]

        if dataset.rag_questions and not gold_questions:
            print(f"Error: Dataset {dataset_path} has no answered questions")
            return

        gold_sources = {q.question_id: q.sources for q in gold_questions}
        pred_sources = {
            r.question_id: r.retrieved_sources for r in student.search_results
        }

        ks = [n for n in DEFAULT_K if n <= student.k]

        too_long = find_too_long_sources(pred_sources)
        for source in too_long:
            first_index = source.first_character_index
            last_index = source.last_character_index
            length = last_index - first_index
            print(
                f"Source {source.file_path}[{first_index}:{last_index}] "
                f"has a length of {length} which is more than the limit "
                f"of {MAX_SOURCE_LENGTH} characters"
            )

        print(f"Student data is valid: {not too_long}")
        if too_long:
            print("Student search results are not valid")
            return

        with_student_sources = sum(
            1 for question_id in gold_sources if pred_sources.get(question_id)
        )
        print(f"Total number of questions: {len(gold_questions)}")
        print(f"Total number of questions with sources: {len(gold_sources)}")
        print(
            "Total number of questions with student sources: "
            f"{with_student_sources}"
        )
        print()

        for k in ks:
            recall = recall_at_k(pred_sources, gold_sources, k)
            print(f"Recall@{k}: {recall:.3f} ({recall * 100:.1f}%)")


if __name__ == "__main__":
    Fire(RagCLI)
