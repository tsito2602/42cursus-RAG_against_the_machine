"""Recall@k evaluation of retrieved sources, compatible with moulinette."""

from typing import TypeAlias

from src.models import MinimalSource

IOU_THRESHOLD = 0.05
MAX_SOURCE_LENGTH = 2000

SourcesById: TypeAlias = dict[str, list[MinimalSource]]


def find_too_long_sources(
    sources_by_id: SourcesById, max_length: int = MAX_SOURCE_LENGTH
) -> list[MinimalSource]:
    """Find retrieved sources longer than the allowed length.

    Args:
        sources_by_id: Retrieved sources keyed by question_id.
        max_length: Maximum number of characters per source.

    Returns:
        The sources longer than max_length.
    """
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
    """Check whether a retrieved source matches a gold source.

    Args:
        pred: Retrieved source.
        gold: Gold source.
        iou_threshold: Minimum IoU of the character spans.

    Returns:
        True if both are in the same file and their IoU is at least
        iou_threshold.
    """
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
    """Compute the mean recall@k over the gold questions.

    Empty gold sources count as 1.0, missing predictions as 0.0.

    Args:
        pred_sources: Retrieved sources keyed by question_id, best first.
        gold_sources: Gold sources keyed by question_id.
        k: Number of top retrieved sources to use.

    Returns:
        The mean recall, or 0.0 if gold_sources is empty.
    """
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
