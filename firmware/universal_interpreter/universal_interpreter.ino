// universal_interpreter_v2.ino  (Arduino IDE version)
// Self-Describing Hot-Swappable Sensor Modules — Universal Inference Engine v2
//
// What's new compared to firmware/universal_interpreter (v1):
//   * Works with two module types, chosen from the EEPROM descriptor:
//       - MPU6050 gesture module   (sensor_type 0x02, I2C)
//       - HC-SR501 PIR module      (sensor_type 0x04, digital output on AOUT = GPIO34)
//   * Hot swap: notices when a module is removed or a different one is plugged in
//     and reloads the model without pressing reset or re-uploading.
//   * Prints swap timings (detect -> model loaded -> first inference) and
//     inference time in microseconds, for the paper.
//   * I2C at 100 kHz with the ESP32 internal pull-ups on, so it works on a
//     breadboard without external pull-up resistors.
//
// Board: "ESP32 Dev Module". Serial Monitor: 115200 baud.

#include <Wire.h>
#include <math.h>
#include "driver/gpio.h"

// ── Pins / bus ──
#define SDA_PIN     21
#define SCL_PIN     22
#define AOUT_PIN    34          // analog/digital sensor output from the module connector
#define I2C_HZ      100000

// ── I2C Addresses ──
#define EEPROM_ADDR 0x50
#define MPU_ADDR    0x68

// ── Descriptor constants ──
#define DESC_MAGIC_0     0x53
#define DESC_MAGIC_1     0x48
#define DESC_MAX_LAYERS  8
#define DESC_MAX_CLASSES 16
#define DESC_LABEL_LEN   16
#define DESC_WEIGHTS_OFF 0x0200
#define DESC_NORM_OFF    0x01A4

// Sensor types
#define SENSOR_AUDIO     0x01
#define SENSOR_ACCEL     0x02   // MPU6050
#define SENSOR_VIBRATION 0x03
#define SENSOR_PIR       0x04   // HC-SR501

// Interface byte (header[15]); 0 in older images = I2C
#define IFACE_I2C   0x00
#define IFACE_AOUT  0x01

// Layer types
#define LAYER_CONV1D     0x01
#define LAYER_MAXPOOL1D  0x02
#define LAYER_DENSE      0x03
#define LAYER_FLATTEN    0x04
#define LAYER_RELU       0x05
#define LAYER_SOFTMAX    0x06

// Activations
#define ACT_NONE    0x00
#define ACT_RELU    0x01
#define ACT_SOFTMAX 0x02

// PIR needs to settle after power-up before its output is meaningful
const uint32_t PIR_WARMUP_MS = 60000;

// ── Parsed descriptor ──
struct Header {
  uint8_t  num_layers;
  uint8_t  num_classes;
  uint8_t  sensor_type;
  uint16_t sample_rate_hz;
  uint16_t window_size;
  uint8_t  num_channels;
  uint8_t  quant_type;
  uint16_t weights_total;
  uint8_t  interface;
};

struct LayerDesc {
  uint8_t  layer_type;
  uint8_t  activation;
  uint16_t input_size;
  uint16_t output_size;
  uint16_t kernel_size;
  uint16_t stride;
  uint16_t num_filters;
  uint16_t weight_offset;
  uint16_t weight_bytes;
};

Header hdr;
LayerDesc layers[DESC_MAX_LAYERS];
char labels[DESC_MAX_CLASSES][DESC_LABEL_LEN];
float norm_means[6];
float norm_stds[6];
uint8_t num_norm;

uint8_t loadedSignature[16];      // header of the module currently loaded
float* weights = nullptr;
bool moduleReady = false;
bool firstInferenceDone = false;
uint32_t tDetect = 0, tLoaded = 0, tSensorReady = 0;

// ── EEPROM helpers ──
bool eepromPresent() {
  Wire.beginTransmission(EEPROM_ADDR);
  return Wire.endTransmission() == 0;
}

bool eepromRead(uint16_t addr, uint8_t* buf, uint16_t len) {
  while (len > 0) {
    uint8_t chunk = min((uint16_t)28, len);  // Wire buffer limit
    Wire.beginTransmission(EEPROM_ADDR);
    Wire.write((uint8_t)(addr >> 8));
    Wire.write((uint8_t)(addr & 0xFF));
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom((int)EEPROM_ADDR, (int)chunk) != chunk) return false;
    for (uint8_t i = 0; i < chunk; i++) buf[i] = Wire.read();
    buf += chunk;
    addr += chunk;
    len -= chunk;
  }
  return true;
}

float eepromReadFloat(uint16_t addr) {
  uint8_t b[4];
  eepromRead(addr, b, 4);
  float val;
  memcpy(&val, b, 4);
  return val;
}

const char* sensorName(uint8_t t) {
  switch (t) {
    case SENSOR_AUDIO:     return "Audio";
    case SENSOR_ACCEL:     return "Accelerometer (MPU6050)";
    case SENSOR_VIBRATION: return "Vibration";
    case SENSOR_PIR:       return "PIR presence (HC-SR501)";
    default:               return "Unknown";
  }
}

// ── Parse descriptor ──
bool parseDescriptor() {
  uint8_t header_buf[16];
  if (!eepromRead(0x0000, header_buf, 16)) { Serial.println("READ ERROR"); return false; }

  if (header_buf[0] != DESC_MAGIC_0 || header_buf[1] != DESC_MAGIC_1) {
    Serial.println("BAD MAGIC");
    return false;
  }
  memcpy(loadedSignature, header_buf, 16);

  hdr.num_layers     = header_buf[3];
  hdr.num_classes    = header_buf[4];
  hdr.sensor_type    = header_buf[5];
  hdr.sample_rate_hz = header_buf[6] | (header_buf[7] << 8);
  hdr.window_size    = header_buf[8] | (header_buf[9] << 8);
  hdr.num_channels   = header_buf[10];
  hdr.quant_type     = header_buf[11];
  hdr.weights_total  = header_buf[12] | (header_buf[13] << 8);
  hdr.interface      = header_buf[15];

  if (hdr.num_layers > DESC_MAX_LAYERS || hdr.num_classes > DESC_MAX_CLASSES ||
      hdr.num_classes == 0 || hdr.sample_rate_hz == 0 || hdr.window_size == 0) {
    Serial.println("DESCRIPTOR OUT OF RANGE");
    return false;
  }

  for (int i = 0; i < hdr.num_layers; i++) {
    uint8_t lb[16];
    eepromRead(0x0010 + i * 16, lb, 16);
    layers[i].layer_type    = lb[0];
    layers[i].activation    = lb[1];
    layers[i].input_size    = lb[2] | (lb[3] << 8);
    layers[i].output_size   = lb[4] | (lb[5] << 8);
    layers[i].kernel_size   = lb[6] | (lb[7] << 8);
    layers[i].stride        = lb[8] | (lb[9] << 8);
    layers[i].num_filters   = lb[10] | (lb[11] << 8);
    layers[i].weight_offset = lb[12] | (lb[13] << 8);
    layers[i].weight_bytes  = lb[14] | (lb[15] << 8);
  }

  for (int i = 0; i < hdr.num_classes; i++) {
    uint8_t lb[DESC_LABEL_LEN];
    eepromRead(0x0090 + i * DESC_LABEL_LEN, lb, DESC_LABEL_LEN);
    memcpy(labels[i], lb, DESC_LABEL_LEN);
    labels[i][DESC_LABEL_LEN - 1] = '\0';
  }

  uint8_t norm_hdr[2];
  eepromRead(0x0190, norm_hdr, 2);
  num_norm = norm_hdr[0];
  if (num_norm > 6) num_norm = 6;
  for (int i = 0; i < num_norm; i++) {
    norm_means[i] = eepromReadFloat(DESC_NORM_OFF + i * 4);
    norm_stds[i]  = eepromReadFloat(DESC_NORM_OFF + num_norm * 4 + i * 4);
  }
  return true;
}

// ── Load weights from EEPROM into RAM ──
bool loadWeights() {
  if (weights) { free(weights); weights = nullptr; }
  uint32_t total = hdr.weights_total;
  weights = (float*)malloc(total);
  if (!weights) { Serial.println("MALLOC FAIL"); return false; }

  Serial.printf("Loading weights: %lu bytes... ", (unsigned long)total);
  if (!eepromRead(DESC_WEIGHTS_OFF, (uint8_t*)weights, total)) {
    Serial.println("READ ERROR");
    return false;
  }
  Serial.println("done.");
  return true;
}

// ── Inference kernels ──
void kern_conv1d(const float* input, int T, int C_in, const float* kernel, const float* bias,
                 int K, int C_out, float* output) {
  int pad = K / 2;
  for (int t = 0; t < T; t++) {
    for (int co = 0; co < C_out; co++) {
      float sum = bias[co];
      for (int k = 0; k < K; k++) {
        int ti = t - pad + k;
        if (ti < 0 || ti >= T) continue;
        for (int ci = 0; ci < C_in; ci++)
          sum += input[ti * C_in + ci] * kernel[k * C_in * C_out + ci * C_out + co];
      }
      output[t * C_out + co] = sum;
    }
  }
}

void kern_maxpool1d(const float* input, int T, int C, int pool, int stride, float* output, int* T_out) {
  *T_out = T / stride;
  for (int t = 0; t < *T_out; t++) {
    for (int c = 0; c < C; c++) {
      float mx = input[(t * stride) * C + c];
      for (int p = 1; p < pool && (t * stride + p) < T; p++) {
        float v = input[(t * stride + p) * C + c];
        if (v > mx) mx = v;
      }
      output[t * C + c] = mx;
    }
  }
}

void kern_dense(const float* input, int in_sz, const float* w, const float* b, float* output, int out_sz) {
  for (int o = 0; o < out_sz; o++) {
    float sum = b[o];
    for (int i = 0; i < in_sz; i++) sum += input[i] * w[i * out_sz + o];
    output[o] = sum;
  }
}

void kern_relu(float* data, int n) { for (int i = 0; i < n; i++) if (data[i] < 0) data[i] = 0; }

void kern_softmax(float* x, int n) {
  float mx = x[0];
  for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
  float sum = 0;
  for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
  for (int i = 0; i < n; i++) x[i] /= sum;
}

// ── Universal inference dispatcher (ping-pong buffers) ──
static float buf_a[4096];
static float buf_b[4096];

void runInference(const float* input, float* scores) {
  int input_len = hdr.window_size * hdr.num_channels;
  memcpy(buf_a, input, input_len * sizeof(float));
  float* src = buf_a;
  float* dst = buf_b;
  int cur_T = hdr.window_size, cur_C = hdr.num_channels, cur_flat = cur_T * cur_C;

  for (int i = 0; i < hdr.num_layers; i++) {
    LayerDesc& L = layers[i];
    switch (L.layer_type) {
      case LAYER_CONV1D: {
        int K = L.kernel_size, C_out = L.num_filters;
        float* kw = (float*)((uint8_t*)weights + L.weight_offset);
        float* kb = kw + K * cur_C * C_out;
        kern_conv1d(src, cur_T, cur_C, kw, kb, K, C_out, dst);
        if (L.activation == ACT_RELU) kern_relu(dst, cur_T * C_out);
        cur_C = C_out; cur_flat = cur_T * cur_C;
        float* tmp = src; src = dst; dst = tmp;
        break;
      }
      case LAYER_MAXPOOL1D: {
        int new_T;
        kern_maxpool1d(src, cur_T, cur_C, L.kernel_size, L.stride, dst, &new_T);
        cur_T = new_T; cur_flat = cur_T * cur_C;
        float* tmp = src; src = dst; dst = tmp;
        break;
      }
      case LAYER_FLATTEN:
        cur_flat = cur_T * cur_C;
        break;
      case LAYER_DENSE: {
        int in_sz = L.input_size, out_sz = L.output_size;
        float* dw = (float*)((uint8_t*)weights + L.weight_offset);
        float* db = dw + in_sz * out_sz;
        kern_dense(src, in_sz, dw, db, dst, out_sz);
        if (L.activation == ACT_RELU) kern_relu(dst, out_sz);
        if (L.activation == ACT_SOFTMAX) kern_softmax(dst, out_sz);
        cur_flat = out_sz; cur_T = 1; cur_C = out_sz;
        float* tmp = src; src = dst; dst = tmp;
        break;
      }
      case LAYER_RELU:    kern_relu(src, cur_flat); break;
      case LAYER_SOFTMAX: kern_softmax(src, cur_flat); break;
    }
  }
  memcpy(scores, src, hdr.num_classes * sizeof(float));
}

// ── Sensor drivers (selected by the descriptor) ──
void initMPU() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);
}

void readIMU(float* v) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, 1);
  v[0] = (int16_t)(Wire.read() << 8 | Wire.read());
  v[1] = (int16_t)(Wire.read() << 8 | Wire.read());
  v[2] = (int16_t)(Wire.read() << 8 | Wire.read());
  Wire.read(); Wire.read();                       // temperature
  v[3] = (int16_t)(Wire.read() << 8 | Wire.read());
  v[4] = (int16_t)(Wire.read() << 8 | Wire.read());
  v[5] = (int16_t)(Wire.read() << 8 | Wire.read());
}

// Returns false if the sensor the descriptor asks for isn't usable
bool initSensor() {
  if (hdr.sensor_type == SENSOR_PIR && hdr.interface == IFACE_AOUT) {
    pinMode(AOUT_PIN, INPUT);
    Serial.printf("PIR on GPIO%d. Warming up %lu s (send 's' to skip)...\n",
                  AOUT_PIN, (unsigned long)(PIR_WARMUP_MS / 1000));
    uint32_t t0 = millis(), lastPrint = 0;
    while (millis() - t0 < PIR_WARMUP_MS) {
      if (Serial.available() && Serial.read() == 's') break;
      if (millis() - lastPrint >= 10000) {
        lastPrint = millis();
        Serial.printf("  %lu s left\n", (unsigned long)((PIR_WARMUP_MS - (millis() - t0)) / 1000));
      }
      if (!eepromPresent()) return false;           // module pulled during warm-up
      delay(20);
    }
    Serial.printf("PIR warm-up took %lu ms\n", (unsigned long)(millis() - t0));
    return true;
  }
  if (hdr.sensor_type == SENSOR_ACCEL) {
    Serial.print("Initializing MPU6050... ");
    Wire.beginTransmission(MPU_ADDR);
    if (Wire.endTransmission() != 0) { Serial.println("NOT FOUND"); return false; }
    initMPU();
    Serial.println("OK");
    return true;
  }
  Serial.printf("No driver for sensor type 0x%02X / interface %d\n", hdr.sensor_type, hdr.interface);
  return false;
}

// Fill one window of input. Returns false if the module disappeared.
bool captureWindow(float* input) {
  uint32_t period_us = 1000000UL / hdr.sample_rate_hz;
  uint32_t next = micros();
  int idx = 0;
  for (int i = 0; i < hdr.window_size; i++) {
    while ((int32_t)(micros() - next) < 0) { }
    next += period_us;
    if (hdr.sensor_type == SENSOR_PIR) {
      input[idx++] = (float)digitalRead(AOUT_PIN);              // raw 0/1, no normalization
    } else {
      float raw[6];
      readIMU(raw);
      for (int j = 0; j < hdr.num_channels && j < 6; j++) {
        float v = raw[j];
        if (j < num_norm && norm_stds[j] != 0) v = (v - norm_means[j]) / norm_stds[j];
        input[idx++] = v;
      }
    }
  }
  return eepromPresent();
}

// ── Dashboard ──
void printDashboard(float* scores, int best, uint32_t infer_us) {
  Serial.println("╔══════════════════════════════════════╗");
  Serial.println("║   SELF-DESCRIBING MODULE — LIVE      ║");
  Serial.println("╠══════════════════════════════════════╣");
  Serial.printf("║ Module:   %-27s║\n", sensorName(hdr.sensor_type));
  Serial.printf("║ Model:    %d layers, %d classes          ║\n", hdr.num_layers, hdr.num_classes);
  Serial.printf("║ Sampling: %d Hz x %d samples          ║\n", hdr.sample_rate_hz, hdr.window_size);
  Serial.printf("║ Inference: %lu us                       ║\n", (unsigned long)infer_us);
  Serial.println("╠══════════════════════════════════════╣");
  for (int i = 0; i < hdr.num_classes; i++) {
    Serial.print(i == best ? "║  ▶ " : "║    ");
    char buf[DESC_LABEL_LEN + 2];
    snprintf(buf, sizeof(buf), "%-10s", labels[i]);
    Serial.print(buf);
    int bar = (int)(scores[i] * 20);
    for (int b = 0; b < 20; b++) Serial.print(b < bar ? "█" : "░");
    Serial.printf(" %5.1f%%║\n", scores[i] * 100);
  }
  Serial.println("╚══════════════════════════════════════╝\n");
}

// ── Module load / unload ──
void unloadModule(const char* why) {
  if (weights) { free(weights); weights = nullptr; }
  moduleReady = false;
  Serial.printf("\n>>> %s. Waiting for a module...\n", why);
}

void loadModule() {
  tDetect = millis();
  Serial.println("\n========================================");
  Serial.println(" Module detected at 0x50");
  Serial.println("========================================");

  Serial.print("Reading descriptor... ");
  if (!parseDescriptor()) { Serial.println("FAILED"); delay(1000); return; }
  Serial.println("OK");

  Serial.println("\n── Module Descriptor ──");
  Serial.printf("  Sensor type:  %s\n", sensorName(hdr.sensor_type));
  Serial.printf("  Interface:    %s\n", hdr.interface == IFACE_AOUT ? "AOUT pin" : "I2C");
  Serial.printf("  Sample rate:  %d Hz\n", hdr.sample_rate_hz);
  Serial.printf("  Window size:  %d\n", hdr.window_size);
  Serial.printf("  Channels:     %d\n", hdr.num_channels);
  Serial.printf("  Layers:       %d\n", hdr.num_layers);
  Serial.printf("  Classes:      %d  (", hdr.num_classes);
  for (int i = 0; i < hdr.num_classes; i++) Serial.printf("%s%s", labels[i], i < hdr.num_classes - 1 ? ", " : ")\n");
  Serial.printf("  Normalization: %s\n", num_norm ? "per-channel mean/std" : "none");

  Serial.println("\n── Layer Architecture ──");
  const char* ltypes[] = {"?", "Conv1D", "MaxPool", "Dense", "Flatten", "ReLU", "Softmax"};
  const char* acts[] = {"none", "relu", "softmax"};
  for (int i = 0; i < hdr.num_layers; i++) {
    int lt = layers[i].layer_type;
    Serial.printf("  [%d] %s", i, lt <= 6 ? ltypes[lt] : "?");
    if (lt == LAYER_CONV1D)    Serial.printf(" k=%d f=%d", layers[i].kernel_size, layers[i].num_filters);
    if (lt == LAYER_MAXPOOL1D) Serial.printf(" k=%d s=%d", layers[i].kernel_size, layers[i].stride);
    if (lt == LAYER_DENSE)     Serial.printf(" %d->%d", layers[i].input_size, layers[i].output_size);
    if (layers[i].activation > 0 && layers[i].activation <= 2) Serial.printf(" +%s", acts[layers[i].activation]);
    if (layers[i].weight_bytes > 0) Serial.printf(" (%dB)", layers[i].weight_bytes);
    Serial.println();
  }
  Serial.println();

  if (!loadWeights()) { delay(1000); return; }
  tLoaded = millis();
  Serial.printf("Descriptor + weights loaded in %lu ms\n", (unsigned long)(tLoaded - tDetect));

  if (!initSensor()) { unloadModule("Sensor not available"); return; }
  tSensorReady = millis();

  Serial.println("\n════════════════════════════════════════");
  Serial.println("  Module ready. Performing inference...");
  Serial.println("════════════════════════════════════════\n");
  moduleReady = true;
  firstInferenceDone = false;
}

// ── Main ──
void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN, I2C_HZ);
  gpio_set_pull_mode((gpio_num_t)SDA_PIN, GPIO_PULLUP_ONLY);   // internal pull-ups
  gpio_set_pull_mode((gpio_num_t)SCL_PIN, GPIO_PULLUP_ONLY);
  delay(500);
  Serial.println("\n========================================");
  Serial.println(" Universal TinyML Inference Engine v2.0");
  Serial.println("========================================");
  Serial.println("Waiting for a module...");
}

void loop() {
  // ── No module loaded: poll for one ──
  if (!moduleReady) {
    if (eepromPresent()) {
      delay(200);                    // let the contacts settle after insertion
      loadModule();
    } else {
      delay(100);
    }
    return;
  }

  // ── Module loaded: has it been removed or swapped? ──
  uint8_t sig[16];
  if (!eepromPresent() || !eepromRead(0x0000, sig, 16)) { unloadModule("Module removed"); return; }
  if (memcmp(sig, loadedSignature, 16) != 0) { unloadModule("Different module detected"); return; }

  // ── Capture a window and classify ──
  int total = hdr.window_size * hdr.num_channels;
  float* input = (float*)malloc(total * sizeof(float));
  if (!input) { Serial.println("INPUT MALLOC FAIL"); delay(1000); return; }
  if (!captureWindow(input)) { free(input); unloadModule("Module removed"); return; }

  float scores[DESC_MAX_CLASSES];
  uint32_t t0 = micros();
  runInference(input, scores);
  uint32_t infer_us = micros() - t0;
  free(input);

  if (!firstInferenceDone) {
    firstInferenceDone = true;
    uint32_t now = millis();
    Serial.println("── Swap timing ──");
    Serial.printf("  detect -> model loaded:      %lu ms\n", (unsigned long)(tLoaded - tDetect));
    Serial.printf("  sensor init / warm-up:       %lu ms\n", (unsigned long)(tSensorReady - tLoaded));
    Serial.printf("  first window + inference:    %lu ms\n", (unsigned long)(now - tSensorReady));
    Serial.printf("  detect -> first inference:   %lu ms\n\n", (unsigned long)(now - tDetect));
  }

  int best = 0;
  for (int i = 1; i < hdr.num_classes; i++) if (scores[i] > scores[best]) best = i;

  if (scores[best] > 0.5f) {
    printDashboard(scores, best, infer_us);
  } else {
    Serial.printf("  ... sampling (%.1f%%)\n", scores[best] * 100);
  }
}
