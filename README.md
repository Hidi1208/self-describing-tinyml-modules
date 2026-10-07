# SDHM — Self-Describing Hot-Swappable TinyML Sensor Modules

A plug-and-play edge AI platform where sensor modules carry their own trained neural network weights and a machine-readable descriptor in onboard EEPROM. A base station (ESP32) reads the descriptor on plug-in, dynamically loads the model, configures the sensor, and begins inference.

## Why This Exists

Every existing edge ML workflow hardcodes the model into firmware. Swap the sensor, reflash the board. SDHM inverts this: the model travels with the sensor. A 512-byte binary descriptor in EEPROM tells the base station everything it needs — sensor type, sampling rate, model architecture (layer types, dimensions, activations), output class labels, and a pointer to the weight blob. A custom universal inference engine parses this descriptor at runtime and executes arbitrary Conv1D/Dense architectures without TensorFlow Lite Micro or any framework dependency.

## Key Numbers

| Metric | Value |
|--------|-------|
| Descriptor size | 512 bytes |
| Model architecture | Conv1D → MaxPool1D → Dense → Dense → Softmax (7 layers) |
| Weight size | 57,296 bytes (INT8 quantized) |
| Gesture classes | idle, shake_x, flick_up, twist |
| Test accuracy | 99.2% |
| Inference target | ESP32 (no PSRAM required) |

## Hardware Setup

![Hardware Setup](images/sdhm-hardware.jpeg)

```
ESP32 GPIO21 (SDA) ──┬── 24LC512 pin 5 ──┬── MPU6050 SDA
ESP32 GPIO22 (SCL) ──┬── 24LC512 pin 6 ──┬── MPU6050 SCL
ESP32 3.3V ──────────┬── 24LC512 pin 8 ──┬── MPU6050 VCC
ESP32 GND ───────────┬── 24LC512 pin 4 ──┬── MPU6050 GND
                      24LC512 pins 1,2,3,7 → GND

4.7kΩ pull-ups on SDA and SCL to 3.3V.
```

## Repo Structure

```
sdhm-tinyml/
├── firmware/
│   ├── eeprom_writer/          # PlatformIO — flashes descriptor + weights to 24LC512
│   └── universal_interpreter/  # PlatformIO — reads EEPROM, runs live inference
├── tools/
│   └── flash_eeprom.py         # Serial handshake script for page-by-page EEPROM write
├── eeprom_image.bin            # Pre-built 64KB EEPROM image (gesture model)
└── README.md
```

## Running the Demo

**Step 1 — Flash the EEPROM (one time):**

```bash
cd firmware/eeprom_writer
pio run -t upload
pip install pyserial
python3 ../../tools/flash_eeprom.py /dev/ttyUSB0 ../../eeprom_image.bin
```

Wait for `DONE errors=0`.

**Step 2 — Upload the interpreter and run:**

```bash
cd ../universal_interpreter
pio run -t upload
pio device monitor
```

![Close-up](images/sdhm-closeup.jpeg)

![Terminal Output](images/sdhm-terminal.jpeg)

**What you'll see:**

1. `Scanning for module EEPROM... FOUND`
2. Full model architecture printout (layer types, dimensions, activations, class labels)
3. `Loading weights: 57296 bytes...`
4. Live gesture classification dashboard with confidence bars

## How It Works

1. On boot, the interpreter scans I2C for a 24LC512 EEPROM
2. Reads the 512-byte descriptor — validates magic bytes and CRC-32
3. Parses model architecture: layer count, types (Conv1D/MaxPool1D/Dense), dimensions, activations
4. Loads INT8 weights from EEPROM into RAM
5. Initializes the sensor (MPU6050) using config from the descriptor
6. Runs inference in a loop — sliding window over accelerometer/gyroscope samples, forward pass through the parsed architecture, softmax output to serial dashboard

The inference engine is a from-scratch C++ implementation — no TFLite Micro, no CMSIS-NN, no external ML framework. It interprets arbitrary layer graphs described by the binary descriptor.

## License

MIT
