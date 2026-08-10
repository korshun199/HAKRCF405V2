from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any


COMMANDS: dict[str, list[str]] = {
    "date": ["date", "--iso-8601=seconds"],
    "uptime": ["uptime"],
    "hostname": ["hostnamectl"],
    "memory": ["free", "-h"],
    "disk": ["df", "-h", "/", "/home/work"],
    "network": ["ip", "-brief", "address"],
    "wifi": ["nmcli", "-f", "GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS", "device", "show", "wlan0"],
    "camera": ["v4l2-ctl", "--list-devices"],
    "vision-status": ["systemctl", "--no-pager", "--full", "status", "robot-vision.service"],
    "vision-log": ["journalctl", "--no-pager", "-u", "robot-vision.service", "-n", "40"],
}


def command_names() -> list[str]:
    return ["help", *COMMANDS]


def run_diagnostic(command: str) -> dict[str, Any]:
    normalized = command.strip().lower()
    if normalized == "help":
        return {"command": normalized, "exit_code": 0, "output": "\n".join(command_names())}
    arguments = COMMANDS.get(normalized)
    if arguments is None:
        return {
            "command": normalized,
            "exit_code": 2,
            "output": f"Команда не разрешена. Доступно: {', '.join(command_names())}",
        }

    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=7, check=False)
        output = (result.stdout + result.stderr).strip()
        return {"command": normalized, "exit_code": result.returncode, "output": output[-20000:]}
    except subprocess.TimeoutExpired:
        return {"command": normalized, "exit_code": 124, "output": "Превышено время выполнения"}
    except OSError as error:
        return {"command": normalized, "exit_code": 127, "output": str(error)}


def system_information() -> dict[str, Any]:
    uptime_seconds = _read_uptime()
    memory = _read_memory()
    disk = shutil.disk_usage("/")
    temperature = _read_temperature()
    load_average = os.getloadavg()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "uptime_seconds": uptime_seconds,
        "memory": memory,
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "load_average": list(load_average),
        "temperature_c": temperature,
    }


def _read_uptime() -> int:
    try:
        return int(float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0]))
    except (OSError, ValueError, IndexError):
        return 0


def _read_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw_value = line.split(":", 1)
            values[name] = int(raw_value.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {"total": 0, "available": 0, "used": 0}

    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    return {"total": total, "available": available, "used": max(0, total - available)}


def _read_temperature() -> float | None:
    temperatures: list[float] = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            value = float(path.read_text(encoding="utf-8").strip())
            temperatures.append(value / 1000 if value > 200 else value)
        except (OSError, ValueError):
            continue
    return round(max(temperatures), 1) if temperatures else None
