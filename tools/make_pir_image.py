#!/usr/bin/env python3
"""
Build a 64 KB EEPROM image for the HC-SR501 PIR module, in the same
descriptor format the universal interpreter already reads (same layout as
eeprom_image.bin for the MPU6050 gesture module).

Usage (from the repo folder):
    python tools\\make_pir_image.py C:\\Users\\sidbr\\Downloads\\pir_presence

Reads   <pir_presence>\\pir_weights.bin and pir_model_meta.json (from export_engine.py)
Writes  pir_eeprom_image.bin  (in the current folder)

Descriptor layout (little-endian):
  0x0000  header (16 B)
            [0..1] magic 'S','H'   [2] version   [3] num_layers   [4] num_classes
            [5] sensor_type        [6..7] sample_rate_hz          [8..9] window_size
            [10] num_channels      [11] quant_type (0 = float32)  [12..13] weights_total
            [14] reserved          [15] interface (0 = I2C, 1 = AOUT pin)
  0x0010  layer table, 16 B per layer (max 8)
  0x0090  class labels, 16 B each (max 16)
  0x0190  num_norm (1 B)   0x01A4  means[num_norm], stds[num_norm] (float32)
  0x0200  weights (float32): for each Dense layer, W[in][out] row-major, then bias[out]
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

EEPROM_SIZE = 65536
WEIGHTS_OFF = 0x0200

SENSOR_PIR = 0x04          # 0x01 audio, 0x02 accelerometer (MPU6050), 0x03 vibration
IFACE_I2C, IFACE_AOUT = 0x00, 0x01
QUANT_FLOAT32 = 0x00

LAYER_DENSE, LAYER_FLATTEN = 0x03, 0x04
ACT_NONE, ACT_SOFTMAX = 0x00, 0x02


def layer_entry(ltype, act, in_sz, out_sz, w_off=0, w_bytes=0):
    # type, act, input, output, kernel, stride, filters, weight_offset, weight_bytes
    return struct.pack("<BBHHHHHHH", ltype, act, in_sz, out_sz, 0, 0, 0, w_off, w_bytes)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../pir_presence")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("pir_eeprom_image.bin")

    meta = json.loads((src / "pir_model_meta.json").read_text())
    blob = (src / "pir_weights.bin").read_bytes()

    dense = next(l for l in meta["layers"] if l["op"] == "Dense")
    n_in, n_out = dense["in"], dense["out"]
    scale = dense["weights"]["scale"]

    # INT8 weights -> float32 (the interpreter's current weight format)
    w_q = np.frombuffer(blob[:n_in * n_out], dtype=np.int8).astype(np.float32)
    W = (w_q * scale).reshape(n_in, n_out)
    b = np.frombuffer(blob[n_in * n_out:n_in * n_out + 4 * n_out], dtype="<f4")

    weights = W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    labels = meta["labels"]

    img = bytearray(b"\xFF" * EEPROM_SIZE)
    img[0:WEIGHTS_OFF] = b"\x00" * WEIGHTS_OFF

    # Header
    header = struct.pack("<2sBBBBHHBBHBB",
                         b"SH", 1,
                         2,                      # layers: Flatten, Dense(+softmax)
                         len(labels),
                         SENSOR_PIR,
                         meta["sample_rate_hz"],
                         meta["window_samples"],
                         meta["channels"],
                         QUANT_FLOAT32,
                         len(weights),
                         0,                      # reserved
                         IFACE_AOUT)
    img[0x0000:0x0010] = header

    # Layer table
    window = meta["window_samples"] * meta["channels"]
    img[0x0010:0x0020] = layer_entry(LAYER_FLATTEN, ACT_NONE, window, window)
    img[0x0020:0x0030] = layer_entry(LAYER_DENSE, ACT_SOFTMAX, n_in, n_out, 0, len(weights))

    # Labels
    for i, name in enumerate(labels):
        raw = name.encode()[:15]
        img[0x0090 + i * 16:0x0090 + (i + 1) * 16] = raw + b"\x00" * (16 - len(raw))

    # Normalization: none (PIR samples are already 0/1)
    img[0x0190] = 0

    # Weights
    img[WEIGHTS_OFF:WEIGHTS_OFF + len(weights)] = weights

    out.write_bytes(img)

    # ── Self-check: parse the image back exactly like the interpreter does ──
    h = img[:16]
    assert h[0:2] == b"SH"
    wt = h[12] | (h[13] << 8)
    Wr = np.frombuffer(img[WEIGHTS_OFF:WEIGHTS_OFF + n_in * n_out * 4], "<f4").reshape(n_in, n_out)
    br = np.frombuffer(img[WEIGHTS_OFF + n_in * n_out * 4:WEIGHTS_OFF + wt], "<f4")
    def predict(x):
        z = x @ Wr + br
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    print(f"Wrote {out}  ({len(img)} bytes)")
    print(f"  sensor: PIR (0x{SENSOR_PIR:02X}), interface: AOUT pin, "
          f"{meta['sample_rate_hz']} Hz x {meta['window_samples']} samples x {meta['channels']} ch")
    print(f"  layers: Flatten -> Dense {n_in}->{n_out} + softmax")
    print(f"  labels: {labels}")
    print(f"  weights: {wt} bytes float32 at 0x{WEIGHTS_OFF:04X}")

    csv = src / "pir_data.csv"
    if csv.exists():
        data = np.loadtxt(csv, delimiter=",", skiprows=1)
        X, y = data[:, :-1].astype(np.float32), data[:, -1].astype(int)
        acc = (predict(X).argmax(1) == y).mean()
        print(f"  check: model read back from the image classifies all {len(y)} recorded "
              f"windows with {acc*100:.1f}% accuracy")
    print("  empty window ->", labels[int(predict(np.zeros(window, np.float32)).argmax())])
    print("  motion window ->", labels[int(predict(np.ones(window, np.float32)).argmax())])


if __name__ == "__main__":
    main()
