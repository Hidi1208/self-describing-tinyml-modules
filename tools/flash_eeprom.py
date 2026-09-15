#!/usr/bin/env python3
import serial, sys, time
from pathlib import Path

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
binfile = sys.argv[2] if len(sys.argv) > 2 else "eeprom_image.bin"

data = Path(binfile).read_bytes()
assert len(data) == 65536

ser = serial.Serial(port, 115200, timeout=10)
time.sleep(2)

while True:
    line = ser.readline().decode(errors="replace").strip()
    print(f"  [{line}]")
    if "READY" in line: break

ser.write(b"WRITE\n")

PAGE = 64
addr = 0
while addr < len(data):
    line = ser.readline().decode(errors="replace").strip()
    if "K" in line:
        ser.write(data[addr:addr+PAGE])
        addr += PAGE
        pct = addr * 100 // len(data)
        print(f"\r  Flashing: {pct}%  ({addr}/{len(data)})", end="", flush=True)
    elif "TIMEOUT" in line or "ERROR" in line:
        print(f"\n  {line}")
        break

print()
for _ in range(10):
    line = ser.readline().decode(errors="replace").strip()
    if line: print(f"  {line}")
    if "DONE" in line: break

ser.write(b"VERIFY\n")
time.sleep(0.5)
for _ in range(5):
    line = ser.readline().decode(errors="replace").strip()
    if line: print(f"  {line}")

ser.close()
print("\nFlash complete.")
