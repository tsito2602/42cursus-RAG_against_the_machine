from fire import Fire
from pydantic import ValidationError

from src.evaluator import (
    MAX_SOURCE_LENGTH,
    find_too_long_sources,
    recall_at_k,
)
from src.models import AnsweredQuestion, RagDataset, StudentSearchResults

DEFAULT_K = (1, 3, 5, 10)


class RagCLI:
    def index(self, max_chunk_size: int = 2000) -> None: ...

    def search(self, query: str, k: int) -> None: ...

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None: ...

    def answer(self, query: str, k: int) -> None: ...

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None: ...

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

        # Duplicate question_ids keep the last entry, as moulinette does
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
