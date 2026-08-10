from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
from flask import Flask, Response, jsonify, request

from config_store import ConfigError, ConfigStore
from system_tools import command_names, run_diagnostic, system_information
from vision import AsyncKettleDetector, KettleDetection, annotate_kettle, find_camera_device


@dataclass
class RuntimeStatus:
    camera: str | None = None
    camera_state: str = "ожидание камеры"
    fps: float = 0.0
    frames: int = 0
    error: str | None = None


class VisionRuntime:
    """Управляет видеозахватом и независимым потоком распознавания."""

    def __init__(self, config_store: ConfigStore) -> None:
        self.config_store = config_store
        self.configuration = config_store.load()
        self.detector: AsyncKettleDetector | None = None
        self.lock = threading.Lock()
        self.reconfigure = threading.Event()
        self.jpeg: bytes | None = None
        self.detection = KettleDetection.empty()
        self.status = RuntimeStatus()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="camera")

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        self.reconfigure.set()
        if self.detector is not None:
            self.detector.stop()
        self.thread.join(timeout=3)

    def snapshot(self) -> tuple[bytes | None, KettleDetection, RuntimeStatus]:
        with self.lock:
            return self.jpeg, self.detection, RuntimeStatus(**asdict(self.status))

    def apply_configuration(self, configuration: dict[str, Any]) -> None:
        with self.lock:
            self.configuration = deepcopy(configuration)
        self.reconfigure.set()

    def _configuration_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return deepcopy(self.configuration)

    def _run(self) -> None:
        while self.running:
            self.reconfigure.clear()
            configuration = self._configuration_snapshot()
            camera_config = configuration["camera"]
            detector = self._build_detector(configuration["kettle"])
            self.detector = detector
            detector.start()
            try:
                self._capture(camera_config, detector)
            finally:
                detector.stop()
                self.detector = None

            if self.running and not self.reconfigure.is_set():
                self.reconfigure.wait(camera_config["reconnect_seconds"])

    def _capture(self, camera_config: dict[str, Any], detector: AsyncKettleDetector) -> None:
        configured_device = camera_config["device"]
        camera_device = find_camera_device() if configured_device == "auto" else configured_device
        if camera_device is None:
            self._set_waiting("камера не подключена")
            return

        capture = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config["width"])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config["height"])
        capture.set(cv2.CAP_PROP_FPS, camera_config["fps"])
        if not capture.isOpened():
            self._set_waiting(f"не удалось открыть {camera_device}")
            capture.release()
            return

        self._set_camera(camera_device)
        frame_count = 0
        period_started = time.monotonic()
        try:
            while self.running and not self.reconfigure.is_set():
                success, frame = capture.read()
                if not success:
                    self._set_waiting(f"потерян видеопоток {camera_device}")
                    return

                detector.submit(frame)
                detection, detector_error = detector.snapshot()
                annotated = annotate_kettle(frame, detection)
                encoded, jpeg = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, camera_config["jpeg_quality"]]
                )
                if not encoded:
                    continue

                frame_count += 1
                now = time.monotonic()
                elapsed = now - period_started
                with self.lock:
                    self.jpeg = jpeg.tobytes()
                    self.detection = detection
                    self.status.frames += 1
                    self.status.error = detector_error
                    if elapsed >= 1.0:
                        self.status.fps = frame_count / elapsed
                        frame_count = 0
                        period_started = now
        finally:
            capture.release()

    @staticmethod
    def _build_detector(configuration: dict[str, Any]) -> AsyncKettleDetector:
        model_path = Path(configuration["model_path"])
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent / model_path
        return AsyncKettleDetector(
            model_path=model_path,
            input_size=configuration["input_size"],
            confidence_threshold=configuration["confidence_threshold"],
            nms_threshold=configuration["nms_threshold"],
            interval_seconds=configuration["inference_interval_seconds"],
        )

    def _set_waiting(self, message: str) -> None:
        with self.lock:
            self.status.camera = None
            self.status.camera_state = "ожидание камеры"
            self.status.error = message
            self.status.fps = 0.0
            self.jpeg = None
            self.detection = KettleDetection.empty()

    def _set_camera(self, camera_device: str) -> None:
        with self.lock:
            self.status.camera = camera_device
            self.status.camera_state = "камера работает"
            self.status.error = None


app = Flask(__name__)
config_store = ConfigStore()
runtime = VisionRuntime(config_store)
runtime.start()


@app.get("/api/status")
def api_status():
    _, detection, status = runtime.snapshot()
    return jsonify({"runtime": asdict(status), "detection": detection.to_dict()})


@app.get("/api/health")
def api_health():
    _, _, status = runtime.snapshot()
    return jsonify({"status": "ok", "camera": status.camera_state})


@app.get("/api/config")
def api_config_get():
    return jsonify(config_store.load())


@app.put("/api/config")
def api_config_put():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Конфигурация должна быть объектом"}), 400
    try:
        configuration = config_store.save(body)
    except ConfigError as error:
        return jsonify({"error": str(error)}), 400
    runtime.apply_configuration(configuration)
    return jsonify(configuration)


@app.get("/api/system")
def api_system():
    return jsonify({"system": system_information(), "commands": command_names()})


@app.post("/api/command")
def api_command():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("command"), str):
        return jsonify({"error": "Требуется строковое поле command"}), 400
    result = run_diagnostic(body["command"])
    return jsonify(result), 200 if result["exit_code"] == 0 else 400


@app.get("/video")
def video():
    return Response(_mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _mjpeg_stream() -> Iterator[bytes]:
    while True:
        jpeg, _, _ = runtime.snapshot()
        if jpeg is None:
            time.sleep(0.2)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.04)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=config_store.load()["web"]["port"], threaded=True)
