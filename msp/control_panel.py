#!/usr/bin/env python3
"""Веб-панель управления OSD и самолётом через UNO Q → iNAV MSP.

Две секции:
1. OSD — ввод текста, отображается на OSD_CRAFT_NAME
2. Управление — слайдеры Roll/Pitch/Yaw/Throttle/AUX1 + кнопки ARM/DISARM

Порт UART открыт постоянно. MSP_SET_RAW_RC отправляется
непрерывно 10 раз в секунду (iNAV требует минимум 5Hz).

ВНИМАНИЕ: для управления самолётом в iNAV должно быть
set receiver_type = MSP
Иначе MSP_SET_RAW_RC игнорируется.

Запуск: python3 control_panel.py
Веб-интерфейс: http://<IP-UNO-Q>:8080

ПРЕДУПРЕЖДЕНИЕ: непрерывное открытие/закрытие UART-порта
при перезагрузке iNAV может вызвать kernel panic на UNO Q.
Не перезагружать iNAV во время работы панели."""
from __future__ import annotations

import os
import termios
import time
import threading
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

PORT = "/dev/ttyACM0"
BAUD = termios.B115200

# ============================================================================
# MSP низкоуровневый слой — порт открыт постоянно
# ============================================================================

def _crc8_msp(data: bytes) -> int:
    """Контрольная сумма MSP v1 (XOR)."""
    c = 0
    for b in data:
        c ^= b
    return c

class MspPort:
    """Обёртка над UART-портом с блокировкой для потокобезопасности."""
    def __init__(self, port: str):
        self.port = port
        self.fd = None
        self.lock = threading.Lock()

    def open(self) -> bool:
        """Открывает порт и держит его открытым."""
        try:
            self.fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY)
            t = termios.tcgetattr(self.fd)
            t[0] = termios.IGNBRK
            t[1] = 0
            t[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            t[3] = 0
            t[4] = BAUD
            t[5] = BAUD
            t[6][termios.VMIN] = 0
            t[6][termios.VTIME] = 1
            termios.tcsetattr(self.fd, termios.TCSANOW, t)
            return True
        except Exception as e:
            print(f"Ошибка открытия порта: {e}")
            self.fd = None
            return False

    def send(self, cmd: int, payload: bytes = b"") -> bool:
        """Отправляет MSP v1 команду (потокобезопасно)."""
        with self.lock:
            if self.fd is None:
                if not self.open():
                    return False
            try:
                msg = bytes([ord("$"), ord("M"), ord("<"), len(payload), cmd]) + payload
                crc = _crc8_msp([len(payload), cmd] + list(payload))
                os.write(self.fd, msg + bytes([crc]))
                return True
            except Exception as e:
                print(f"Ошибка отправки: {e}")
                self.fd = None
                return False

msp = MspPort(PORT)

# ============================================================================
# API: OSD (craft name)
# ============================================================================

MSP_SET_NAME = 11

def set_osd_text(text: str) -> dict:
    """Устанавливает текст на OSD через craft name."""
    text_bytes = text.encode("ascii", errors="replace").upper()[:16]
    if msp.send(MSP_SET_NAME, text_bytes):
        return {"ok": True, "text": text_bytes.decode("ascii")}
    return {"ok": False, "error": "Ошибка отправки"}

# ============================================================================
# Управление самолётом — непрерывная отправка MSP_SET_RAW_RC
# ============================================================================

MSP_SET_RAW_RC = 200
MID = 1500
MIN_US = 1000
MAX_US = 2000

# Текущее состояние каналов (AETR: A=roll, E=pitch, T=throttle, R=yaw)
_rc_channels = [MID, MID, MIN_US, MID, MIN_US] + [MIN_US] * 11
_rc_lock = threading.Lock()
_rc_running = True

def _rc_loop():
    """Фоновый поток — отправляет MSP_SET_RAW_RC 10 раз в секунду."""
    while _rc_running:
        with _rc_lock:
            channels = list(_rc_channels)
        payload = b""
        for ch in channels:
            payload += ch.to_bytes(2, "little")
        msp.send(MSP_SET_RAW_RC, payload)
        time.sleep(0.1)  # 10 Hz

_rc_thread = threading.Thread(target=_rc_loop, daemon=True, name="rc-sender")
_rc_thread.start()

def update_rc(channels: list[int]) -> dict:
    """Обновляет значения каналов."""
    while len(channels) < 16:
        channels.append(MIN_US)
    channels = channels[:16]
    channels = [max(MIN_US, min(MAX_US, ch)) for ch in channels]
    with _rc_lock:
        _rc_channels[:] = channels
    return {"ok": True, "channels": channels}

# ============================================================================
# Flask маршруты
# ============================================================================

@app.route("/")
def index():
    return send_file("/home/work/HAKRCF405V2/msp/control_panel.html")

@app.route("/api/osd", methods=["POST"])
def api_osd():
    body = request.get_json(silent=True)
    if not body or "text" not in body:
        return jsonify({"error": "Нет поля text"}), 400
    return jsonify(set_osd_text(str(body["text"])))

@app.route("/api/rc", methods=["POST"])
def api_rc():
    body = request.get_json(silent=True)
    if not body or "channels" not in body:
        return jsonify({"error": "Нет поля channels"}), 400
    return jsonify(update_rc(list(body["channels"])))

@app.route("/api/rc/neutral", methods=["POST"])
def api_rc_neutral():
    """Нейтральная позиция — A=1500, E=1500, T=1000, R=1500, AUX=1000."""
    channels = [MID, MID, MIN_US, MID] + [MIN_US] * 12
    return jsonify(update_rc(channels))

@app.route("/api/rc/arm", methods=["POST"])
def api_rc_arm():
    """Арм — AUX1=1800, throttle низкий."""
    channels = [MID, MID, MIN_US, MID, 1800] + [MIN_US] * 11
    return jsonify(update_rc(channels))

@app.route("/api/rc/disarm", methods=["POST"])
def api_rc_disarm():
    """Дисарм — AUX1=1000."""
    channels = [MID, MID, MIN_US, MID, 1000] + [MIN_US] * 11
    return jsonify(update_rc(channels))

@app.route("/api/rc/steer", methods=["POST"])
def api_rc_steer():
    """Управление: roll(A), pitch(E), yaw(R), throttle(T), aux1."""
    body = request.get_json(silent=True) or {}
    roll = int(body.get("roll", MID))
    pitch = int(body.get("pitch", MID))
    yaw = int(body.get("yaw", MID))
    throttle = int(body.get("throttle", MIN_US))
    aux1 = int(body.get("aux1", MIN_US))
    channels = [roll, pitch, throttle, yaw, aux1] + [MIN_US] * 11
    return jsonify(update_rc(channels))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
