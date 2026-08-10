from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_store import ConfigError, ConfigStore, DEFAULT_CONFIG, validate_config


class ConfigStoreTest(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        validated = validate_config(deepcopy(DEFAULT_CONFIG))

        self.assertEqual(validated["camera"]["width"], 640)
        self.assertEqual(validated["vision"]["threshold"], 28)

    def test_partial_config_uses_defaults(self) -> None:
        validated = validate_config({"vision": {"threshold": 42}})

        self.assertEqual(validated["vision"]["threshold"], 42)
        self.assertEqual(validated["camera"]["fps"], 25)
        self.assertEqual(validated["kettle"]["model_path"], "models/kettle_480.onnx")
        self.assertEqual(validated["kettle"]["input_size"], 480)

    def test_invalid_value_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            validate_config({"camera": {"width": 10}})

    def test_store_saves_and_loads_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            store = ConfigStore(path)
            configuration = deepcopy(DEFAULT_CONFIG)
            configuration["vision"]["threshold"] = 35

            saved = store.save(configuration)
            loaded = store.load()

            self.assertEqual(saved, loaded)
            self.assertEqual(loaded["vision"]["threshold"], 35)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["vision"]["threshold"], 35)


if __name__ == "__main__":
    unittest.main()
