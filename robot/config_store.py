from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "camera": {
        "device": "auto",
        "width": 640,
        "height": 480,
        "fps": 25,
        "jpeg_quality": 82,
        "reconnect_seconds": 2,
    },
    "vision": {
        "background_alpha": 0.025,
        "threshold": 28,
        "minimum_area_ratio": 0.006,
        "history_size": 8,
        "horizontal_threshold": 0.025,
        "approach_ratio": 1.14,
        "recede_ratio": 0.88,
    },
    "kettle": {
        "model_path": "models/kettle_480.onnx",
        "input_size": 480,
        "confidence_threshold": 0.35,
        "nms_threshold": 0.45,
        "inference_interval_seconds": 0.5,
    },
    "web": {"port": 8000, "status_refresh_ms": 500, "system_refresh_ms": 5000},
}


class ConfigError(ValueError):
    pass


class ConfigStore:
    def __init__(self, path: Path | None = None) -> None:
        configured_path = os.getenv("ROBOT_CONFIG")
        self.path = path or (Path(configured_path) if configured_path else Path(__file__).with_name("config.json"))
        self.lock = Lock()

    def load(self) -> dict[str, Any]:
        with self.lock:
            return self._load_unlocked()

    def save(self, value: dict[str, Any]) -> dict[str, Any]:
        validated = validate_config(value)
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
                    json.dump(validated, config_file, ensure_ascii=False, indent=2)
                    config_file.write("\n")
                os.replace(temporary_path, self.path)
            except Exception:
                Path(temporary_path).unlink(missing_ok=True)
                raise
        return deepcopy(validated)

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return deepcopy(DEFAULT_CONFIG)
        try:
            with self.path.open(encoding="utf-8") as config_file:
                value = json.load(config_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigError(f"Не удалось прочитать {self.path}: {error}") from error
        if not isinstance(value, dict):
            raise ConfigError("Корень конфигурации должен быть объектом")
        return validate_config(value)


def validate_config(value: dict[str, Any]) -> dict[str, Any]:
    merged = _deep_merge(DEFAULT_CONFIG, value)
    camera = merged.get("camera")
    vision = merged.get("vision")
    kettle = merged.get("kettle")
    web = merged.get("web")
    if not all(isinstance(section, dict) for section in (camera, vision, kettle, web)):
        raise ConfigError("Разделы camera, vision, kettle и web должны быть объектами")

    device = camera.get("device")
    if not isinstance(device, str) or not device.strip():
        raise ConfigError("camera.device должен быть непустой строкой")
    camera["device"] = device.strip()
    camera["width"] = _integer(camera.get("width"), "camera.width", 160, 3840)
    camera["height"] = _integer(camera.get("height"), "camera.height", 120, 2160)
    camera["fps"] = _integer(camera.get("fps"), "camera.fps", 1, 120)
    camera["jpeg_quality"] = _integer(camera.get("jpeg_quality"), "camera.jpeg_quality", 30, 100)
    camera["reconnect_seconds"] = _number(camera.get("reconnect_seconds"), "camera.reconnect_seconds", 0.2, 60)

    vision["background_alpha"] = _number(vision.get("background_alpha"), "vision.background_alpha", 0.001, 0.5)
    vision["threshold"] = _integer(vision.get("threshold"), "vision.threshold", 1, 255)
    vision["minimum_area_ratio"] = _number(vision.get("minimum_area_ratio"), "vision.minimum_area_ratio", 0.0001, 0.5)
    vision["history_size"] = _integer(vision.get("history_size"), "vision.history_size", 4, 120)
    vision["horizontal_threshold"] = _number(vision.get("horizontal_threshold"), "vision.horizontal_threshold", 0.001, 0.5)
    vision["approach_ratio"] = _number(vision.get("approach_ratio"), "vision.approach_ratio", 1.01, 10)
    vision["recede_ratio"] = _number(vision.get("recede_ratio"), "vision.recede_ratio", 0.05, 0.99)

    model_path = kettle.get("model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ConfigError("kettle.model_path должен быть непустой строкой")
    kettle["model_path"] = model_path.strip()
    kettle["input_size"] = _integer(kettle.get("input_size"), "kettle.input_size", 32, 2048)
    kettle["confidence_threshold"] = _number(
        kettle.get("confidence_threshold"), "kettle.confidence_threshold", 0.01, 1.0
    )
    kettle["nms_threshold"] = _number(kettle.get("nms_threshold"), "kettle.nms_threshold", 0.01, 1.0)
    kettle["inference_interval_seconds"] = _number(
        kettle.get("inference_interval_seconds"), "kettle.inference_interval_seconds", 0.05, 60.0
    )

    web["port"] = _integer(web.get("port"), "web.port", 1024, 65535)
    web["status_refresh_ms"] = _integer(web.get("status_refresh_ms"), "web.status_refresh_ms", 100, 60000)
    web["system_refresh_ms"] = _integer(web.get("system_refresh_ms"), "web.system_refresh_ms", 500, 60000)
    return merged


def _deep_merge(original: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(original)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ConfigError(f"{name} должен быть целым числом от {minimum} до {maximum}")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= float(value) <= maximum:
        raise ConfigError(f"{name} должен быть числом от {minimum} до {maximum}")
    return float(value)
