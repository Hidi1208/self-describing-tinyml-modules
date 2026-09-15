#include <Arduino.h>
#include <Wire.h>

#define EEPROM_ADDR 0x50
#define EEPROM_SIZE 65536
#define PAGE_SIZE 64

void writeEEPROM(uint16_t addr, const uint8_t* data, uint8_t len) {
  Wire.beginTransmission(EEPROM_ADDR);
  Wire.write((uint8_t)(addr >> 8));
  Wire.write((uint8_t)(addr & 0xFF));
  for (uint8_t i = 0; i < len; i++) Wire.write(data[i]);
  Wire.endTransmission();
  delay(5);
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
  Wire.begin(21, 22);
  Wire.setClock(400000);
  delay(500);

  Wire.beginTransmission(EEPROM_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println("ERROR: No EEPROM at 0x50");
    while(1);
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
      Serial.println("K");  // request next page

      size_t got = 0;
      unsigned long t0 = millis();
      while (got < PAGE_SIZE && millis() - t0 < 10000) {
        if (Serial.available()) {
          page[got++] = Serial.read();
          t0 = millis();
        }
      }
      if (got < PAGE_SIZE) {
        Serial.print("TIMEOUT at 0x");
        Serial.println(addr, HEX);
        return;
      }

      // Write in 32-byte chunks
      for (int c = 0; c < PAGE_SIZE; c += 32) {
        writeEEPROM(addr + c, page + c, 32);
      }
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
    uint8_t m0 = readEEPROM(0);
    uint8_t m1 = readEEPROM(1);
    Serial.println(m0 == 0x53 && m1 == 0x48 ? "MAGIC: OK" : "MAGIC: FAIL");
  }
}
