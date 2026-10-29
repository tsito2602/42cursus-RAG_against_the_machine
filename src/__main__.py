from fire import Fire
from pydantic import ValidationError

from src.evaluator import find_too_long_sources, recall_at_k
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
        try:
            with open(student_search_results_path, encoding="utf-8") as f:
                student = StudentSearchResults.model_validate_json(f.read())
            with open(dataset_path, encoding="utf-8") as f:
                dataset = RagDataset.model_validate_json(f.read())
        except (OSError, ValidationError) as e:
            print(f"Error: {e}")
            return

        answers = [
            a for a in dataset.rag_questions if isinstance(a, AnsweredQuestion)
        ]

        if not answers:
            print(f"Error: Dataset {dataset_path} has no answered questions.")
            return

        results = {
            r.question_id: r.retrieved_sources for r in student.search_results
        }

        ks = [n for n in DEFAULT_K if n <= student.k]

        too_long = find_too_long_sources(results)
        for source in too_long:
            first_index = source.first_character_index
            last_index = source.last_character_index
            length = last_index - first_index
            print(
                f"Source {source.file_path}[{first_index}:{last_index}] "
                f"has a length of {length} which is more than the limit "
                "of 2000 characters"
            )

        print(f"Student data is valid: {not too_long}")
        if too_long:
            print("Student search results are not valid")
            return

        print(f"Total number of questions: {len(answers)}")
        with_sources = sum(1 for q in answers if results.get(q.question_id))
        print(
            f"Total number of questions with student sources: {with_sources}"
        )
        print()

        for k in ks:
            recall = recall_at_k(results, answers, k)
            print(f"Recall@{k}: {recall:.3f} ({recall * 100:.1f}%)")


if __name__ == "__main__":
    Fire(RagCLI)
