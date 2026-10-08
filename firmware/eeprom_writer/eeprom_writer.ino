// eeprom_writer_arduino.ino  (Arduino IDE version of firmware/eeprom_writer)
// Receives a 64 KB image from tools/flash_eeprom.py and writes it to the 24LC512.
// Same serial protocol as the PlatformIO version; changes:
//   * 100 kHz I2C with ESP32 internal pull-ups (works without external resistors)
//   * ACK polling after each write instead of a fixed 5 ms wait
//   * VERIFY also prints the sensor type, so you can see which module the chip holds
//
// Board: "ESP32 Dev Module". Close the Serial Monitor before running flash_eeprom.py.

#include <Wire.h>
#include "driver/gpio.h"

#define EEPROM_ADDR 0x50
#define EEPROM_SIZE 65536
#define PAGE_SIZE   64

bool waitWriteDone() {
  for (int i = 0; i < 20; i++) {
    Wire.beginTransmission(EEPROM_ADDR);
    if (Wire.endTransmission() == 0) return true;
    delay(1);
  }
  return false;
}

void writeEEPROM(uint16_t addr, const uint8_t* data, uint8_t len) {
  Wire.beginTransmission(EEPROM_ADDR);
  Wire.write((uint8_t)(addr >> 8));
  Wire.write((uint8_t)(addr & 0xFF));
  for (uint8_t i = 0; i < len; i++) Wire.write(data[i]);
  Wire.endTransmission();
  waitWriteDone();
}

uint8_t readEEPROM(uint16_t addr) {
  Wire.beginTransmission(EEPROM_ADDR);
  Wire.write((uint8_t)(addr >> 8));
  Wire.write((uint8_t)(addr & 0xFF));
  Wire.endTransmission(false);
  Wire.requestFrom(EEPROM_ADDR, 1);
  return Wire.read();
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22, 100000);
  gpio_set_pull_mode(GPIO_NUM_21, GPIO_PULLUP_ONLY);
  gpio_set_pull_mode(GPIO_NUM_22, GPIO_PULLUP_ONLY);
  delay(500);

  Wire.beginTransmission(EEPROM_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println("ERROR: No EEPROM at 0x50");
    while (1) delay(1000);
  }
  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();

  if (cmd == "WRITE") {
    uint32_t addr = 0;
    uint8_t page[PAGE_SIZE];
    while (addr < EEPROM_SIZE) {
      Serial.println("K");                     // request next page
      size_t got = 0;
      unsigned long t0 = millis();
      while (got < PAGE_SIZE && millis() - t0 < 10000) {
        if (Serial.available()) { page[got++] = Serial.read(); t0 = millis(); }
      }
      if (got < PAGE_SIZE) {
        Serial.print("TIMEOUT at 0x");
        Serial.println(addr, HEX);
        return;
      }
      for (int c = 0; c < PAGE_SIZE; c += 32) writeEEPROM(addr + c, page + c, 32);
      addr += PAGE_SIZE;
    }
    Serial.println("DONE");

  } else if (cmd == "VERIFY") {
    Serial.print("HEADER: ");
    for (int i = 0; i < 16; i++) {
      uint8_t b = readEEPROM(i);
      if (b < 16) Serial.print("0");
      Serial.print(b, HEX);
      Serial.print(" ");
    }
    Serial.println();
    uint8_t m0 = readEEPROM(0), m1 = readEEPROM(1);
    Serial.println(m0 == 0x53 && m1 == 0x48 ? "MAGIC: OK" : "MAGIC: FAIL");
    uint8_t st = readEEPROM(5);
    Serial.print("SENSOR: ");
    Serial.println(st == 0x02 ? "MPU6050 gesture module" : st == 0x04 ? "PIR presence module" : "unknown");
  }
}
