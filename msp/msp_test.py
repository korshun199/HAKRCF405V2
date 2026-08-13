#!/usr/bin/env python3
"""Тест связи с полётным контроллером по MSP.
Отправляет MSP_IDENT (1) и проверяет ответ."""
import os, sys, termios, time, select

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
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

print(f'Открываю {PORT}')
fd = open_port(PORT)
time.sleep(0.3)

# MSP_IDENT (1)
payload = b''
msg = bytes([ord('$'), ord('M'), ord('<'), len(payload), 1]) + payload
crc = crc8_msp([len(payload), 1] + list(payload))
os.write(fd, msg + bytes([crc]))

print(f'Отправляю: {msg.hex()}{crc:02x}')

end = time.time() + 2
out = b''
while time.time() < end:
    if select.select([fd], [], [], 0.05)[0]:
        try:
            out += os.read(fd, 1024)
        except OSError:
            break
        if b'$M>' in out:
            break

print(f'Получено: {out!r}')
print(f'Hex: {out.hex() if out else "None"}')

if b'$M>' in out:
    print('OK: полётный контроллер ответил по MSP')
else:
    print('ОШИБКА: нет ответа')

os.close(fd)
