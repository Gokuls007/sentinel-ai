"""Desk posture coach (front-facing laptop webcam, compared with your own upright baseline)."""

from posture.coach import (
    BAD,
    GOOD,
    LABELS,
    LEANING,
    SLOUCHING,
    TOO_CLOSE,
    Baseline,
    Measurement,
    PostureCoach,
    PostureConfig,
    classify,
    measure,
)

__all__ = ["BAD", "GOOD", "LABELS", "LEANING", "SLOUCHING", "TOO_CLOSE", "Baseline", "Measurement", "PostureCoach",
           "PostureConfig", "classify", "measure"]
