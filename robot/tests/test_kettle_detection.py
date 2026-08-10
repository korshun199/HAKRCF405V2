from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision import AsyncKettleDetector, KettleDetection, classify_horizontal, decode_kettle_output


class KettleOutputTest(unittest.TestCase):
    def test_decodes_channels_first_output(self) -> None:
        output = np.array(
            [
                [
                    [240.0, 100.0, 400.0],
                    [180.0, 100.0, 300.0],
                    [120.0, 40.0, 20.0],
                    [80.0, 40.0, 20.0],
                    [0.92, 0.20, 0.70],
                ]
            ],
            dtype=np.float32,
        )

        detection = decode_kettle_output(
            output=output,
            frame_shape=(360, 640),
            scale=0.75,
            padding=(0, 105),
            confidence_threshold=0.35,
            nms_threshold=0.45,
            inference_seconds=0.123,
        )

        self.assertTrue(detection.detected)
        self.assertEqual(detection.label, "электрочайник")
        self.assertAlmostEqual(detection.confidence, 0.92, places=5)
        self.assertEqual(detection.box, (240, 47, 160, 106))
        self.assertEqual(detection.center, (320, 100))
        self.assertEqual(detection.horizontal, "по центру")
        self.assertEqual(detection.inference_seconds, 0.123)

    def test_accepts_predictions_first_output(self) -> None:
        output = np.array([[[96.0, 240.0, 48.0, 96.0, 0.8]]], dtype=np.float32)

        detection = decode_kettle_output(
            output=output,
            frame_shape=(480, 480),
            scale=1.0,
            padding=(0, 0),
            confidence_threshold=0.5,
            nms_threshold=0.45,
        )

        self.assertTrue(detection.detected)
        self.assertEqual(detection.horizontal, "слева")

    def test_returns_empty_result_below_threshold(self) -> None:
        output = np.array([[[240.0], [240.0], [100.0], [100.0], [0.24]]], dtype=np.float32)

        detection = decode_kettle_output(
            output=output,
            frame_shape=(480, 480),
            scale=1.0,
            padding=(0, 0),
            confidence_threshold=0.25,
            nms_threshold=0.45,
            inference_seconds=0.05,
        )

        self.assertFalse(detection.detected)
        self.assertEqual(detection.label, "электрочайник")
        self.assertIsNone(detection.box)
        self.assertEqual(detection.inference_seconds, 0.05)

    def test_classifies_horizontal_position(self) -> None:
        self.assertEqual(classify_horizontal(100, 640), "слева")
        self.assertEqual(classify_horizontal(320, 640), "по центру")
        self.assertEqual(classify_horizontal(540, 640), "справа")

    def test_serializes_structured_result(self) -> None:
        detection = KettleDetection(True, "электрочайник", 0.8, (10, 20, 30, 40), (25, 40), "слева", 0.2)

        value = detection.to_dict()

        self.assertEqual(value["box"], {"x": 10, "y": 20, "width": 30, "height": 40})
        self.assertEqual(value["center"], {"x": 25, "y": 40})


class FakeKettleDetector:
    def __init__(self, **_: object) -> None:
        self.started = threading.Event()

    def infer(self, frame: np.ndarray) -> KettleDetection:
        self.started.set()
        time.sleep(0.05)
        return KettleDetection(True, "электрочайник", 0.9, (1, 2, 3, 4), (2, 4), "слева", 0.05)


class AsyncKettleDetectorTest(unittest.TestCase):
    def test_processes_frame_in_worker_without_waiting(self) -> None:
        detector = AsyncKettleDetector(
            model_path=Path("unused.onnx"),
            input_size=480,
            confidence_threshold=0.35,
            nms_threshold=0.45,
            interval_seconds=0.1,
            detector_factory=FakeKettleDetector,
        )
        detector.start()
        try:
            started = time.monotonic()
            accepted = detector.submit(np.zeros((32, 32, 3), dtype=np.uint8))
            submission_seconds = time.monotonic() - started
            self.assertTrue(accepted)
            self.assertLess(submission_seconds, 0.04)

            deadline = time.monotonic() + 1.0
            detection = KettleDetection.empty()
            while time.monotonic() < deadline:
                detection, error = detector.snapshot()
                if detection.detected:
                    break
                time.sleep(0.01)
            self.assertTrue(detection.detected)
            self.assertIsNone(error)
        finally:
            detector.stop()


if __name__ == "__main__":
    unittest.main()
