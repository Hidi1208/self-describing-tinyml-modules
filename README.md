# SDHM Demo — Step by Step

## Wiring
ESP32 GPIO21 (SDA) ──┬── EEPROM pin 5 ──┬── MPU6050 SDA
ESP32 GPIO22 (SCL) ──┬── EEPROM pin 6 ──┬── MPU6050 SCL
ESP32 3.3V ──────────┬── EEPROM pin 8 ──┬── MPU6050 VCC
ESP32 GND ───────────┬── EEPROM pin 4 ──┬── MPU6050 GND
                      EEPROM pins 1,2,3,7 → GND
4.7kΩ pullup SDA→3.3V, 4.7kΩ pullup SCL→3.3V

## Step 1: Flash EEPROM (one time)
```
cd eeprom_writer
pio run -t upload
pip install pyserial
python3 ../flash_eeprom.py /dev/ttyUSB0 ../eeprom_image.bin
```
Wait for "DONE errors=0"

## Step 2: Upload interpreter
```
cd ../universal_interpreter
pio run -t upload
pio device monitor
```

## What you'll see
1. "Scanning for module EEPROM... FOUND"
2. Full model architecture printout (layers, classes, labels)
3. "Loading weights: 57296 bytes..."
4. Live inference dashboard with gesture classification
