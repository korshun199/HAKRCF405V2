from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from system_tools import command_names, run_diagnostic, system_information


class SystemToolsTest(unittest.TestCase):
    def test_help_lists_available_commands(self) -> None:
        result = run_diagnostic("help")

        self.assertEqual(result["exit_code"], 0)
        self.assertIn("memory", result["output"])
        self.assertIn("vision-status", result["output"])

    def test_unknown_command_is_rejected(self) -> None:
        result = run_diagnostic("rm -rf /")

        self.assertEqual(result["exit_code"], 2)
        self.assertIn("не разрешена", result["output"])

    def test_command_names_are_unique(self) -> None:
        names = command_names()

        self.assertEqual(len(names), len(set(names)))

    def test_system_information_has_expected_fields(self) -> None:
        information = system_information()

        self.assertIn("hostname", information)
        self.assertIn("memory", information)
        self.assertIn("disk", information)
        self.assertGreater(information["disk"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
