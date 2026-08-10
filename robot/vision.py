from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    detected: bool
    horizontal: str
    depth: str
    center_x: int | None
    center_y: int | None
    width: int | None
    height: int | None
    area_ratio: float
    foreground_ratio: float

    def to_dict(self) -> dict[str, bool | str | int | float | None]:
        return asdict(self)


class MotionDetector:
    def __init__(
        self,
        background_alpha: float = 0.025,
        threshold: int = 28,
        minimum_area_ratio: float = 0.006,
        history_size: int = 8,
        horizontal_threshold: float = 0.025,
        approach_ratio: float = 1.14,
        recede_ratio: float = 0.88,
    ) -> None:
        self.background_alpha = background_alpha
        self.threshold = threshold
        self.minimum_area_ratio = minimum_area_ratio
        self.horizontal_threshold = horizontal_threshold
        self.approach_ratio = approach_ratio
        self.recede_ratio = recede_ratio
        self.background: np.ndarray | None = None
        self.history: deque[tuple[float, float, float]] = deque(maxlen=history_size)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def reset(self) -> None:
        self.background = None
        self.history.clear()

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self.background is None:
            self.background = gray.astype(np.float32)
            return frame.copy(), self._empty_detection()

        background_u8 = cv2.convertScaleAbs(self.background)
        difference = cv2.absdiff(gray, background_u8)
        _, mask = cv2.threshold(difference, self.threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        frame_area = frame.shape[0] * frame.shape[1]
        minimum_area = frame_area * self.minimum_area_ratio
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
        foreground_ratio = float(cv2.countNonZero(mask) / frame_area)

        background_mask = cv2.bitwise_not(mask)
        cv2.accumulateWeighted(gray, self.background, self.background_alpha, mask=background_mask)

        if not contours:
            self.history.clear()
            return frame.copy(), self._empty_detection(foreground_ratio)

        contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(contour)
        center_x = x + width // 2
        center_y = y + height // 2
        area_ratio = float((width * height) / frame_area)
        now = monotonic()
        self.history.append((now, center_x / frame.shape[1], area_ratio))
        horizontal, depth = self._classify_motion()

        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (80, 220, 120), 2)
        cv2.circle(annotated, (center_x, center_y), 4, (50, 150, 255), -1)
        cv2.putText(
            annotated,
            f"{horizontal}; {depth}",
            (max(8, x), max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (80, 220, 120),
            2,
            cv2.LINE_AA,
        )

        return annotated, Detection(
            detected=True,
            horizontal=horizontal,
            depth=depth,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            area_ratio=area_ratio,
            foreground_ratio=foreground_ratio,
        )

    def _classify_motion(self) -> tuple[str, str]:
        if len(self.history) < 4:
            return "ожидание", "ожидание"

        _, start_x, start_area = self.history[0]
        _, end_x, end_area = self.history[-1]
        horizontal_delta = end_x - start_x
        if horizontal_delta > self.horizontal_threshold:
            horizontal = "движется вправо"
        elif horizontal_delta < -self.horizontal_threshold:
            horizontal = "движется влево"
        else:
            horizontal = "по центру"

        area_change = end_area / max(start_area, 1e-6)
        if area_change > self.approach_ratio:
            depth = "приближается"
        elif area_change < self.recede_ratio:
            depth = "удаляется"
        else:
            depth = "дистанция стабильна"

        return horizontal, depth

    @staticmethod
    def _empty_detection(foreground_ratio: float = 0.0) -> Detection:
        return Detection(
            detected=False,
            horizontal="объект не обнаружен",
            depth="объект не обнаружен",
            center_x=None,
            center_y=None,
            width=None,
            height=None,
            area_ratio=0.0,
            foreground_ratio=foreground_ratio,
        )


@dataclass(frozen=True)
class KettleDetection:
    detected: bool
    label: str
    confidence: float
    box: tuple[int, int, int, int] | None
    center: tuple[int, int] | None
    horizontal: str
    inference_seconds: float

    def to_dict(self) -> dict[str, object]:
        box = None
        if self.box is not None:
            box = {"x": self.box[0], "y": self.box[1], "width": self.box[2], "height": self.box[3]}
        center = None
        if self.center is not None:
            center = {"x": self.center[0], "y": self.center[1]}
        return {
            "detected": self.detected,
            "label": self.label,
            "confidence": self.confidence,
            "box": box,
            "center": center,
            "horizontal": self.horizontal,
            "inference_seconds": self.inference_seconds,
        }

    @staticmethod
    def empty(inference_seconds: float = 0.0) -> "KettleDetection":
        return KettleDetection(
            detected=False,
            label="электрочайник",
            confidence=0.0,
            box=None,
            center=None,
            horizontal="не обнаружен",
            inference_seconds=inference_seconds,
        )


def classify_horizontal(center_x: int, frame_width: int) -> str:
    """Определяет положение центра объекта по горизонтали кадра."""
    ratio = center_x / max(frame_width, 1)
    if ratio < 0.4:
        return "слева"
    if ratio > 0.6:
        return "справа"
    return "по центру"


def decode_kettle_output(
    output: np.ndarray,
    frame_shape: tuple[int, int],
    scale: float,
    padding: tuple[int, int],
    confidence_threshold: float,
    nms_threshold: float,
    inference_seconds: float = 0.0,
) -> KettleDetection:
    """Декодирует выход YOLO и возвращает лучшее обнаружение чайника."""
    predictions = np.asarray(output, dtype=np.float32).squeeze()
    if predictions.ndim == 1 and predictions.size >= 5:
        predictions = predictions.reshape(1, -1)
    if predictions.ndim != 2:
        raise ValueError(f"Ожидался двумерный выход модели, получена форма {predictions.shape}")
    if predictions.shape[0] == 5 and predictions.shape[1] != 5:
        predictions = predictions.T
    elif predictions.shape[0] > 5 and predictions.shape[1] > predictions.shape[0]:
        predictions = predictions.T
    if predictions.shape[1] < 5:
        raise ValueError(f"В выходе модели должно быть не менее 5 значений, получено {predictions.shape[1]}")

    scores = predictions[:, 4:].max(axis=1)
    selected = predictions[scores >= confidence_threshold]
    selected_scores = scores[scores >= confidence_threshold]
    if selected.size == 0:
        return KettleDetection.empty(inference_seconds)

    frame_height, frame_width = frame_shape
    pad_x, pad_y = padding
    boxes: list[list[int]] = []
    confidences: list[float] = []
    for row, score in zip(selected, selected_scores):
        center_x, center_y, width, height = (float(value) for value in row[:4])
        left = int(round((center_x - width / 2 - pad_x) / scale))
        top = int(round((center_y - height / 2 - pad_y) / scale))
        right = int(round((center_x + width / 2 - pad_x) / scale))
        bottom = int(round((center_y + height / 2 - pad_y) / scale))
        left = min(max(left, 0), frame_width - 1)
        top = min(max(top, 0), frame_height - 1)
        right = min(max(right, 0), frame_width)
        bottom = min(max(bottom, 0), frame_height)
        box_width = right - left
        box_height = bottom - top
        if box_width <= 0 or box_height <= 0:
            continue
        boxes.append([left, top, box_width, box_height])
        confidences.append(float(score))

    if not boxes:
        return KettleDetection.empty(inference_seconds)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    if len(indices) == 0:
        return KettleDetection.empty(inference_seconds)
    flattened = np.asarray(indices).reshape(-1)
    best_index = max((int(index) for index in flattened), key=lambda index: confidences[index])
    x, y, width, height = boxes[best_index]
    center = (x + width // 2, y + height // 2)
    return KettleDetection(
        detected=True,
        label="электрочайник",
        confidence=confidences[best_index],
        box=(x, y, width, height),
        center=center,
        horizontal=classify_horizontal(center[0], frame_width),
        inference_seconds=inference_seconds,
    )


class KettleDetector:
    """Выполняет распознавание чайника через OpenCV DNN."""

    def __init__(
        self,
        model_path: Path,
        input_size: int = 480,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
    ) -> None:
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.network = cv2.dnn.readNetFromONNX(str(model_path))
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def infer(self, frame: np.ndarray) -> KettleDetection:
        started = monotonic()
        prepared, scale, padding = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(
            prepared,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        self.network.setInput(blob)
        output = self.network.forward()
        elapsed = monotonic() - started
        return decode_kettle_output(
            output=output,
            frame_shape=frame.shape[:2],
            scale=scale,
            padding=padding,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            inference_seconds=elapsed,
        )

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        height, width = frame.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2
        prepared = cv2.copyMakeBorder(
            resized,
            pad_y,
            self.input_size - resized_height - pad_y,
            pad_x,
            self.input_size - resized_width - pad_x,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return prepared, scale, (pad_x, pad_y)


class AsyncKettleDetector:
    """Хранит только свежий кадр и выполняет тяжёлый вывод в отдельном потоке."""

    def __init__(
        self,
        model_path: Path,
        input_size: int,
        confidence_threshold: float,
        nms_threshold: float,
        interval_seconds: float,
        detector_factory: Callable[..., KettleDetector] = KettleDetector,
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.interval_seconds = interval_seconds
        self.detector_factory = detector_factory
        self.condition = threading.Condition()
        self.pending_frame: np.ndarray | None = None
        self.latest = KettleDetection.empty()
        self.last_submission = float("-inf")
        self.busy = False
        self.running = False
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="kettle-detector")

    def start(self) -> None:
        self.running = True
        self.thread.start()

    def stop(self) -> None:
        with self.condition:
            self.running = False
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=3)

    def submit(self, frame: np.ndarray) -> bool:
        now = monotonic()
        with self.condition:
            if not self.running or self.busy or self.pending_frame is not None:
                return False
            if now - self.last_submission < self.interval_seconds:
                return False
            self.pending_frame = frame.copy()
            self.last_submission = now
            self.condition.notify()
            return True

    def snapshot(self) -> tuple[KettleDetection, str | None]:
        with self.condition:
            return self.latest, self.error

    def _run(self) -> None:
        try:
            detector = self.detector_factory(
                model_path=self.model_path,
                input_size=self.input_size,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
            )
        except Exception as error:
            with self.condition:
                self.error = f"Не удалось загрузить модель: {error}"
                self.running = False
            return

        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending_frame is not None or not self.running)
                if not self.running:
                    return
                frame = self.pending_frame
                self.pending_frame = None
                self.busy = True
            try:
                if frame is not None:
                    result = detector.infer(frame)
                    with self.condition:
                        self.latest = result
                        self.error = None
            except Exception as error:
                with self.condition:
                    self.error = f"Ошибка распознавания: {error}"
            finally:
                with self.condition:
                    self.busy = False


def annotate_kettle(frame: np.ndarray, detection: KettleDetection) -> np.ndarray:
    """Рисует рамку и латинскую подпись, совместимую с cv2.putText."""
    annotated = frame.copy()
    if not detection.detected or detection.box is None or detection.center is None:
        return annotated
    x, y, width, height = detection.box
    color = (80, 220, 120)
    cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
    cv2.circle(annotated, detection.center, 4, (50, 150, 255), -1)
    cv2.putText(
        annotated,
        f"Electric kettle {detection.confidence * 100:.1f}%",
        (max(8, x), max(24, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated


def find_camera_device() -> str | None:
    for video_path in sorted(Path("/dev").glob("video*")):
        name_path = Path("/sys/class/video4linux") / video_path.name / "name"
        device_name = name_path.read_text(encoding="utf-8").strip().lower() if name_path.exists() else ""
        if any(value in device_name for value in ("decoder", "encoder", "venus")):
            continue
        return str(video_path)
    return None
