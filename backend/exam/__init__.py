"""Exam Hall mode (docs/plans/exam-hall.md): sessions, seats, calibration, later signals."""

from exam.calibration import Baseline, SeatCalibration, hand_height
from exam.head_pose import HeadPose, HeadPoseEstimator, KeypointHeadPose, default_estimator
from exam.monitor import ExamConfig, ExamMonitor
from exam.seats import Seat, assign, label_seats, neighbour_graph, relabel, seat_rect_for
from exam.setup_check import SetupCheck

__all__ = ["Baseline", "ExamConfig", "ExamMonitor", "HeadPose", "HeadPoseEstimator", "KeypointHeadPose",
           "Seat", "SeatCalibration", "SetupCheck", "assign", "default_estimator", "hand_height", "label_seats",
           "neighbour_graph", "relabel", "seat_rect_for"]
