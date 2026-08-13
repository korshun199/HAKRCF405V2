#!/usr/bin/env python3
"""Вывод текста на OSD полётного контроллера через MSP_SET_NAME.
Текст появляется в элементе OSD_CRAFT_NAME.
Максимум 16 символов: A-Z, 0-9, пробел."""
import os, sys, termios, time

PORT = '/dev/ttyACM0'
BAUD = termios.B115200
MSP_SET_NAME = 11  # Команда установки craft name

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

# Текст из аргумента командной строки
text = sys.argv[1] if len(sys.argv) > 1 else 'NONE 0'
text = text.encode('ascii', errors='replace').upper()[:16]

fd = open_port(PORT)
time.sleep(0.2)
msp_send(fd, MSP_SET_NAME, text)
time.sleep(0.1)
os.close(fd)
print('OSD:', text.decode('ascii'))
