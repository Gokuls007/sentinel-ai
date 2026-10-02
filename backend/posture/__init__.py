"""Desk posture coach (front-facing laptop webcam, compared with your own upright baseline)."""

from posture.coach import (
    BAD,
    GOOD,
    LABELS,
    LEANING,
    MOVED,
    SLOUCHING,
    SLUMPED,
    TOO_CLOSE,
    Baseline,
    Measurement,
    PostureCoach,
    PostureConfig,
    assess,
    classify,
    explain,
    measure,
    select_main_person,
    shoulder_problem,
)

__all__ = [
    "BAD", "GOOD", "LABELS", "LEANING", "MOVED", "SLOUCHING", "SLUMPED", "TOO_CLOSE", "Baseline", "Measurement",
    "PostureCoach", "PostureConfig", "assess", "classify", "explain", "measure", "select_main_person",
    "shoulder_problem",
]
