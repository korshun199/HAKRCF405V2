from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision import MotionDetector


class MotionDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.background = np.full((240, 320, 3), 32, dtype=np.uint8)

    def test_detects_motion_to_the_right(self) -> None:
        detector = MotionDetector(background_alpha=0.01)
        detector.process(self.background)

        detection = None
        for x in (40, 58, 76, 94):
            frame = self.background.copy()
            cv2.rectangle(frame, (x, 90), (x + 44, 140), (230, 230, 230), -1)
            _, detection = detector.process(frame)

        self.assertIsNotNone(detection)
        self.assertTrue(detection.detected)
        self.assertEqual(detection.horizontal, "движется вправо")

    def test_detects_motion_to_the_left(self) -> None:
        detector = MotionDetector(background_alpha=0.01)
        detector.process(self.background)

        detection = None
        for x in (190, 170, 150, 130):
            frame = self.background.copy()
            cv2.rectangle(frame, (x, 90), (x + 44, 140), (230, 230, 230), -1)
            _, detection = detector.process(frame)

        self.assertIsNotNone(detection)
        self.assertTrue(detection.detected)
        self.assertEqual(detection.horizontal, "движется влево")

    def test_detects_approaching_object(self) -> None:
        detector = MotionDetector(background_alpha=0.01)
        detector.process(self.background)

        detection = None
        for size in (28, 34, 42, 52):
            frame = self.background.copy()
            x = 160 - size // 2
            y = 120 - size // 2
            cv2.rectangle(frame, (x, y), (x + size, y + size), (230, 230, 230), -1)
            _, detection = detector.process(frame)

        self.assertIsNotNone(detection)
        self.assertTrue(detection.detected)
        self.assertEqual(detection.depth, "приближается")

    def test_detects_receding_object(self) -> None:
        detector = MotionDetector(background_alpha=0.01)
        detector.process(self.background)

        detection = None
        for size in (54, 46, 38, 30):
            frame = self.background.copy()
            x = 160 - size // 2
            y = 120 - size // 2
            cv2.rectangle(frame, (x, y), (x + size, y + size), (230, 230, 230), -1)
            _, detection = detector.process(frame)

        self.assertIsNotNone(detection)
        self.assertTrue(detection.detected)
        self.assertEqual(detection.depth, "удаляется")

    def test_reports_no_object_on_background(self) -> None:
        detector = MotionDetector()
        detector.process(self.background)
        _, detection = detector.process(self.background.copy())

        self.assertFalse(detection.detected)
        self.assertEqual(detection.horizontal, "объект не обнаружен")


if __name__ == "__main__":
    unittest.main()
