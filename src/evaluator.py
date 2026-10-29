from typing import TypeAlias

from src.models import MinimalSource

IOU_THRESHOLD = 0.05
MAX_SOURCE_LENGTH = 2000

SourcesById: TypeAlias = dict[str, list[MinimalSource]]


def find_too_long_sources(
    sources_by_id: SourcesById, max_length: int = MAX_SOURCE_LENGTH
) -> list[MinimalSource]:
    return [
        source
        for sources in sources_by_id.values()
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
    pred_sources: SourcesById,
    gold_sources: SourcesById,
    k: int,
) -> float:
    if len(gold_sources) == 0:
        return 0.0

    hits = 0.0
    for question_id, golds in gold_sources.items():
        if not golds:
            # Match moulinette:
            # questions with no gold sources count as full recall
            hits += 1.0
            continue
        preds = pred_sources.get(question_id, [])[:k]
        found = sum(
            any(overlaps(pred, gold) for pred in preds) for gold in golds
        )
        hits += found / len(golds)

    return hits / len(gold_sources)
