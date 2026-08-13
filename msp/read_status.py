#!/usr/bin/env python3
"""Чтение текущих RC-каналов и активных режимов iNAV.
Диагностический скрипт: показывает значения каналов и какие режимы включены."""
import os, termios, time, select

PORT = '/dev/ttyACM0'
BAUD = termios.B115200

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

def msp1_send(fd, cmd, payload=b''):
    msg = bytes([ord('$'), ord('M'), ord('<'), len(payload), cmd]) + payload
    crc = crc8_msp([len(payload), cmd] + list(payload))
    os.write(fd, msg + bytes([crc]))
    time.sleep(0.1)

def read_all(fd, timeout=2.0):
    end = time.time() + timeout
    out = b''
    while time.time() < end:
        if select.select([fd], [], [], 0.05)[0]:
            try:
                out += os.read(fd, 1024)
            except OSError:
                break
            if b'$M>' in out:
                time.sleep(0.05)
    return out

def parse_v1_all(out):
    results = []
    idx = 0
    while True:
        pos = out.find(b'$M>', idx)
        if pos == -1:
            break
        length = out[pos+3]
        cmd = out[pos+4]
        payload = out[pos+5:pos+5+length]
        results.append((cmd, payload))
        idx = pos + 5 + length + 1
    return results

fd = open_port(PORT)
time.sleep(0.5)

# MSP_BOXIDS (119) — ID режимов
msp1_send(fd, 119, b'')
out = read_all(fd)
for cmd, payload in parse_v1_all(out):
    if cmd == 119:
        print('MSP_BOXIDS — ID режимов:')
        for i, b in enumerate(payload):
            print(f'  Box {i}: id={b}')

# MSP_BOX (113) — активные режимы
msp1_send(fd, 113, b'')
out = read_all(fd)
for cmd, payload in parse_v1_all(out):
    if cmd == 113:
        print('\nMSP_BOX — активные режимы:')
        for i, b in enumerate(payload):
            if b:
                print(f'  byte {i}: 0x{b:02x} = {bin(b)}')

# MSP_RC (105) — текущие каналы
msp1_send(fd, 105, b'')
out = read_all(fd)
for cmd, payload in parse_v1_all(out):
    if cmd == 105:
        print('\nMSP_RC — текущие каналы:')
        for i in range(len(payload)//2):
            ch = int.from_bytes(payload[i*2:i*2+2], 'little')
            print(f'  CH{i+1}: {ch}')

os.close(fd)
