from typing import TypeAlias
from src.models import AnsweredQuestion, MinimalSource

IOU_THRESHOLD = 0.05
MAX_SOURCE_LENGTH = 2000

Results: TypeAlias = dict[str, list[MinimalSource]]


def find_too_long_sources(
    results: Results, max_length: int = MAX_SOURCE_LENGTH
) -> list[MinimalSource]:
    return [
        source
        for sources in results.values()
        for source in sources
        if source.last_character_index - source.first_character_index
        > max_length
    ]


def overlaps(
    pred: MinimalSource,
    gold: MinimalSource,
    iou_threshold: float = IOU_THRESHOLD,
) -> bool:
    if pred.file_path != gold.file_path:
        return False

    intersection = min(
        pred.last_character_index, gold.last_character_index
    ) - max(pred.first_character_index, gold.first_character_index)

    if intersection <= 0:
        return False

    union = max(pred.last_character_index, gold.last_character_index) - min(
        pred.first_character_index, gold.first_character_index
    )

    return intersection / union >= iou_threshold


def recall_at_k(
    results: Results,
    answers: list[AnsweredQuestion],
    k: int,
) -> float:
    if len(answers) == 0:
        return 0.0

    hits = 0.0
    for q in answers:
        if not q.sources:
            # Match moulinette:
            # questions with no gold sources count as full recall
            hits += 1.0
            continue
        preds = results.get(q.question_id, [])[:k]
        found = sum(
            any(overlaps(pred, gold) for pred in preds) for gold in q.sources
        )
        hits += found / len(q.sources)

    return hits / len(answers)
