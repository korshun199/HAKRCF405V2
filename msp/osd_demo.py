#!/usr/bin/env python3
"""Демо: перебор текстовых надписей на OSD.
Каждые N секунд меняет текст на OSD через MSP_SET_NAME.
Запуск: python3 osd_demo.py
Остановка: Ctrl+C

ВНИМАНИЕ: не запускать несколько копий одновременно —
текст будет мерцать как гирлянда."""
import os, termios, time

PORT = '/dev/ttyACM0'
BAUD = termios.B115200
MSP_SET_NAME = 11

def crc8_msp(data):
    c = 0
    for b in data:
        c ^= b
    return c

def open_port(port):
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY)
    t = termios.tcgetattr(fd)
    t[0] = termios.IGNBRK
    t[1] = 0
    t[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    t[3] = 0
    t[4] = BAUD
    t[5] = BAUD
    t[6][termios.VMIN] = 0
    t[6][termios.VTIME] = 1
    termios.tcsetattr(fd, termios.TCSANOW, t)
    return fd

def msp_send(fd, cmd, payload=b''):
    msg = bytes([ord('$'), ord('M'), ord('<'), len(payload), cmd]) + payload
    crc = crc8_msp([len(payload), cmd] + list(payload))
    os.write(fd, msg + bytes([crc]))

# Список надписей для перебора
NAMES = ['TRAKTOR', 'VENIK', 'STAKAN', 'CHAYNIK']
DELAY = 15.0  # секунд между сменой

fd = open_port(PORT)
time.sleep(0.3)
i = 0
print('Демо запущено. Ctrl+C для остановки.')
try:
    while True:
        name = NAMES[i % len(NAMES)]
        num = i % 6  # 0..5
        text = f'{name} {num}'.encode('ascii')
        msp_send(fd, MSP_SET_NAME, text)
        print('OSD:', text.decode('ascii'))
        i += 1
        time.sleep(DELAY)
except KeyboardInterrupt:
    print('\nСтоп.')
finally:
    os.close(fd)
