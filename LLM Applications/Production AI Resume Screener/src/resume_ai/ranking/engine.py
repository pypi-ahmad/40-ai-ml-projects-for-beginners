"""Candidate ranking and comparison utilities."""

from __future__ import annotations

from resume_ai.models import ScoreBreakdown


def rank_scores(scores: list[ScoreBreakdown]) -> list[ScoreBreakdown]:
    """Sort candidates by total score descending."""
    return sorted(scores, key=lambda item: item.total_score, reverse=True)


def compare_candidates(scores: list[ScoreBreakdown], candidate_ids: list[int]) -> list[ScoreBreakdown]:
    """Filter and sort selected candidates."""
    selected = [score for score in scores if score.candidate_id in set(candidate_ids)]
    return rank_scores(selected)
