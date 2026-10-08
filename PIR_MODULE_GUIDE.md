# Adding the PIR module and demonstrating the hot swap

Files for the PIR module (the two firmware sketches replace the original PlatformIO versions and are opened in Arduino IDE):

| File | What it is |
|---|---|
| `tools/make_pir_image.py` | Builds `pir_eeprom_image.bin` (64 KB) from the trained PIR model |
| `firmware/eeprom_writer/eeprom_writer.ino` | Arduino IDE version of the EEPROM writer (100 kHz, no resistors needed) |
| `firmware/universal_interpreter/universal_interpreter.ino` | Arduino IDE interpreter that runs **both** modules and detects swaps |

## A. Build the PIR EEPROM image (laptop, once)

```
cd C:\Users\sidbr\Downloads\self-describing-tinyml-modules-main
python tools\make_pir_image.py C:\Users\sidbr\Downloads\pir_presence
```

Expected output ends with a line like:
`check: model read back from the image classifies all 300 recorded windows with 9x.x% accuracy`

## B. Write the image to an EEPROM chip

This **overwrites the whole chip**. Use your second 24LC512 for the PIR module. If you only have one chip, you can put the gesture model back later with `eeprom_image.bin`.

1. Wire the EEPROM: pins 1, 2, 3, 4, 7 to GND, pin 8 to 3.3V, pin 5 to GPIO21, pin 6 to GPIO22.
2. **Arduino IDE:** open `firmware\eeprom_writer\eeprom_writer.ino` and upload it. **Close the Serial Monitor.**
3. **Command Prompt** (replace COM8 with your port):
   ```
   python tools\flash_eeprom.py COM8 pir_eeprom_image.bin
   ```
   It takes about 1–2 minutes. At the end you should see `MAGIC: OK` and `SENSOR: PIR presence module`.

To write the gesture model to a chip instead: `python tools\flash_eeprom.py COM8 eeprom_image.bin`

## C. Run the interpreter

1. Wire both sensors to the ESP32 and leave them connected:
   - MPU6050: VCC to 3.3V, GND to GND, SDA to GPIO21, SCL to GPIO22
   - PIR: VCC to VIN, OUT to GPIO34, GND to GND
2. **Arduino IDE:** open `firmware\universal_interpreter\universal_interpreter.ino`. Set the board to **ESP32 Dev Module**, upload, and open the Serial Monitor at **115200**.
3. Plug in one EEPROM chip. The interpreter reads it and prints the descriptor and the layer list, then starts classifying.
   - PIR chip: it waits 60 s for the PIR to warm up. If the PIR has already been powered for more than a minute, type `s` in the Serial Monitor and press Enter to skip.

## D. The hot-swap demo (the paper's main result)

1. Gesture chip in: the dashboard shows idle / shake_x / flick_up / twist.
2. Pull the chip out: it prints `Module removed. Waiting for a module...`
3. Push the PIR chip in: it prints the PIR descriptor and then person / no_person.
4. No reset, no re-upload. The same ESP32 firmware runs both models.

After each insertion it prints a timing block. Copy these numbers into Table I:

```
── Swap timing ──
  detect -> model loaded:      ... ms
  sensor init / warm-up:       ... ms   (PIR: report the warm-up separately)
  first window + inference:    ... ms
  detect -> first inference:   ... ms
```

Each dashboard also shows `Inference: ... us`, the time for one forward pass.

Take a photo of the setup and a screenshot of the Serial Monitor during a swap for the paper.

## Notes

- Model weights are stored as **float32** in the EEPROM (gesture: 57,296 bytes; PIR: 408 bytes). The descriptor's quant byte is 0 for the PIR image.
- Descriptor byte 15 is the **interface byte**: 0 = I²C (older images already have 0 there), 1 = AOUT pin.
- I²C runs at 100 kHz with the ESP32's internal pull-ups. With real 4.7 kΩ resistors fitted, you can raise `I2C_HZ` to 400000 for faster model loading.
