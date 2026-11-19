"""Object detection. Only detection/yoloe.py imports ultralytics; import it explicitly."""

from stereovision.detection.base import Detection, Detector
from stereovision.detection.null import NullDetector

__all__ = ["Detection", "Detector", "NullDetector"]
