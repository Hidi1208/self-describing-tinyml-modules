#include <Arduino.h>
// universal_interpreter.ino
// Self-Describing Hot-Swappable Sensor Module — Universal Inference Engine
// Reads binary descriptor + weights from 24LC512 EEPROM, runs inference on MPU6050 data

#include <Wire.h>
#include <math.h>

// ── I2C Addresses ──
#define EEPROM_ADDR 0x50
#define MPU_ADDR    0x68

// ── Descriptor constants (must match descriptor_format.h) ──
#define DESC_MAGIC_0     0x53
#define DESC_MAGIC_1     0x48
#define DESC_MAX_LAYERS  8
#define DESC_MAX_CLASSES 16
#define DESC_LABEL_LEN   16
#define DESC_WEIGHTS_OFF 0x0200
#define DESC_NORM_OFF    0x01A4

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

// ── Weight storage ──
float* weights = nullptr;  // all weights loaded into PSRAM/heap

// ── EEPROM read functions ──
void eepromRead(uint16_t addr, uint8_t* buf, uint16_t len) {
  while (len > 0) {
    uint8_t chunk = min((uint16_t)28, len);  // Wire buffer limit
    Wire.beginTransmission(EEPROM_ADDR);
    Wire.write((uint8_t)(addr >> 8));
    Wire.write((uint8_t)(addr & 0xFF));
    Wire.endTransmission(false);
    Wire.requestFrom((int)EEPROM_ADDR, (int)chunk);
    for (uint8_t i = 0; i < chunk; i++) {
      buf[i] = Wire.read();
    }
    buf += chunk;
    addr += chunk;
    len -= chunk;
  }
}

float eepromReadFloat(uint16_t addr) {
  uint8_t b[4];
  eepromRead(addr, b, 4);
  float val;
  memcpy(&val, b, 4);
  return val;
}

// ── Parse descriptor ──
bool parseDescriptor() {
  uint8_t header_buf[16];
  eepromRead(0x0000, header_buf, 16);

  // Check magic
  if (header_buf[0] != DESC_MAGIC_0 || header_buf[1] != DESC_MAGIC_1) {
    Serial.println("BAD MAGIC");
    return false;
  }

  hdr.num_layers    = header_buf[3];
  hdr.num_classes   = header_buf[4];
  hdr.sensor_type   = header_buf[5];
  hdr.sample_rate_hz = header_buf[6] | (header_buf[7] << 8);
  hdr.window_size   = header_buf[8] | (header_buf[9] << 8);
  hdr.num_channels  = header_buf[10];
  hdr.quant_type    = header_buf[11];
  hdr.weights_total = header_buf[12] | (header_buf[13] << 8);

  // Read CRC (we'll skip verification for speed, magic is enough for demo)

  // Read layer descriptors
  for (int i = 0; i < hdr.num_layers && i < DESC_MAX_LAYERS; i++) {
    uint8_t lb[16];
    eepromRead(0x0010 + i * 16, lb, 16);
    layers[i].layer_type   = lb[0];
    layers[i].activation   = lb[1];
    layers[i].input_size   = lb[2] | (lb[3] << 8);
    layers[i].output_size  = lb[4] | (lb[5] << 8);
    layers[i].kernel_size  = lb[6] | (lb[7] << 8);
    layers[i].stride       = lb[8] | (lb[9] << 8);
    layers[i].num_filters  = lb[10] | (lb[11] << 8);
    layers[i].weight_offset = lb[12] | (lb[13] << 8);
    layers[i].weight_bytes = lb[14] | (lb[15] << 8);
  }

  // Read class labels
  for (int i = 0; i < hdr.num_classes && i < DESC_MAX_CLASSES; i++) {
    uint8_t lb[DESC_LABEL_LEN];
    eepromRead(0x0090 + i * DESC_LABEL_LEN, lb, DESC_LABEL_LEN);
    memcpy(labels[i], lb, DESC_LABEL_LEN);
    labels[i][DESC_LABEL_LEN - 1] = '\0';
  }

  // Read normalization params
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
  uint32_t total = hdr.weights_total;
  if (total > 60000) {
    // Use PSRAM if available, else heap
    weights = (float*)ps_malloc(total);
    if (!weights) weights = (float*)malloc(total);
  } else {
    weights = (float*)malloc(total);
  }

  if (!weights) {
    Serial.println("MALLOC FAIL");
    return false;
  }

  Serial.print("Loading weights: ");
  Serial.print(total);
  Serial.println(" bytes...");

  uint8_t* dst = (uint8_t*)weights;
  uint16_t addr = DESC_WEIGHTS_OFF;
  uint32_t remaining = total;

  while (remaining > 0) {
    uint16_t chunk = min((uint32_t)28, remaining);
    eepromRead(addr, dst, chunk);
    dst += chunk;
    addr += chunk;
    remaining -= chunk;
  }

  Serial.println("Weights loaded.");
  return true;
}

// ── Inference kernels (from your existing inference_engine.h) ──

inline float relu_f(float x) { return x > 0 ? x : 0; }

void kern_conv1d(const float* input, int T, int C_in,
                 const float* kernel, const float* bias,
                 int K, int C_out, float* output) {
  int pad = K / 2;
  for (int t = 0; t < T; t++) {
    for (int co = 0; co < C_out; co++) {
      float sum = bias[co];
      for (int k = 0; k < K; k++) {
        int ti = t - pad + k;
        if (ti < 0 || ti >= T) continue;
        for (int ci = 0; ci < C_in; ci++) {
          sum += input[ti * C_in + ci] * kernel[k * C_in * C_out + ci * C_out + co];
        }
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

void kern_dense(const float* input, int in_sz,
                const float* w, const float* b,
                float* output, int out_sz) {
  for (int o = 0; o < out_sz; o++) {
    float sum = b[o];
    for (int i = 0; i < in_sz; i++) {
      sum += input[i] * w[i * out_sz + o];
    }
    output[o] = sum;
  }
}

void kern_relu(float* data, int n) {
  for (int i = 0; i < n; i++) if (data[i] < 0) data[i] = 0;
}

void kern_softmax(float* x, int n) {
  float mx = x[0];
  for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
  float sum = 0;
  for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
  for (int i = 0; i < n; i++) x[i] /= sum;
}

// ── Universal inference dispatcher ──

// Working buffers (statically allocated for ESP32)
static float buf_a[4096];  // ping
static float buf_b[4096];  // pong

void runInference(const float* input, float* scores) {
  // Copy input to buf_a
  int input_len = hdr.window_size * hdr.num_channels;
  memcpy(buf_a, input, input_len * sizeof(float));

  float* src = buf_a;
  float* dst = buf_b;

  // Track current tensor dimensions
  int cur_T = hdr.window_size;
  int cur_C = hdr.num_channels;
  int cur_flat = cur_T * cur_C;

  for (int i = 0; i < hdr.num_layers; i++) {
    LayerDesc& L = layers[i];
    float* w_base = weights;

    switch (L.layer_type) {
      case LAYER_CONV1D: {
        int K = L.kernel_size;
        int C_out = L.num_filters;
        float* kw = (float*)((uint8_t*)w_base + L.weight_offset);
        // Bias is right after kernel weights
        int kernel_floats = K * cur_C * C_out;
        float* kb = kw + kernel_floats;

        kern_conv1d(src, cur_T, cur_C, kw, kb, K, C_out, dst);

        // Apply activation
        int out_len = cur_T * C_out;
        if (L.activation == ACT_RELU) kern_relu(dst, out_len);

        cur_C = C_out;
        cur_flat = cur_T * cur_C;
        float* tmp = src; src = dst; dst = tmp;
        break;
      }

      case LAYER_MAXPOOL1D: {
        int pool = L.kernel_size;
        int stride = L.stride;
        int new_T;
        kern_maxpool1d(src, cur_T, cur_C, pool, stride, dst, &new_T);
        cur_T = new_T;
        cur_flat = cur_T * cur_C;
        float* tmp = src; src = dst; dst = tmp;
        break;
      }

      case LAYER_FLATTEN: {
        // No-op, data is already flat in memory
        cur_flat = cur_T * cur_C;
        break;
      }

      case LAYER_DENSE: {
        int in_sz = L.input_size;
        int out_sz = L.output_size;
        float* dw = (float*)((uint8_t*)w_base + L.weight_offset);
        int w_floats = in_sz * out_sz;
        float* db = dw + w_floats;

        kern_dense(src, in_sz, dw, db, dst, out_sz);

        if (L.activation == ACT_RELU) kern_relu(dst, out_sz);
        if (L.activation == ACT_SOFTMAX) kern_softmax(dst, out_sz);

        cur_flat = out_sz;
        cur_T = 1;
        cur_C = out_sz;
        float* tmp = src; src = dst; dst = tmp;
        break;
      }

      case LAYER_RELU:
        kern_relu(src, cur_flat);
        break;

      case LAYER_SOFTMAX:
        kern_softmax(src, cur_flat);
        break;
    }
  }

  // Copy final output to scores
  memcpy(scores, src, hdr.num_classes * sizeof(float));
}

// ── MPU6050 ──
void initMPU() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);
}

void readIMU(float* ax, float* ay, float* az, float* gx, float* gy, float* gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, 1);
  *ax = (int16_t)(Wire.read() << 8 | Wire.read());
  *ay = (int16_t)(Wire.read() << 8 | Wire.read());
  *az = (int16_t)(Wire.read() << 8 | Wire.read());
  Wire.read(); Wire.read(); // temp
  *gx = (int16_t)(Wire.read() << 8 | Wire.read());
  *gy = (int16_t)(Wire.read() << 8 | Wire.read());
  *gz = (int16_t)(Wire.read() << 8 | Wire.read());
}

// ── Dashboard ──
void printDashboard(float* scores, int best) {
  Serial.println("╔══════════════════════════════════════╗");
  Serial.println("║   SELF-DESCRIBING MODULE — LIVE      ║");
  Serial.println("╠══════════════════════════════════════╣");

  Serial.print("║ Module:  ");
  switch (hdr.sensor_type) {
    case 0x01: Serial.print("AUDIO       "); break;
    case 0x02: Serial.print("ACCELEROMETER"); break;
    case 0x03: Serial.print("VIBRATION   "); break;
    default:   Serial.print("UNKNOWN     "); break;
  }
  Serial.println("         ║");

  Serial.print("║ Model:   ");
  Serial.print(hdr.num_layers);
  Serial.print(" layers, ");
  Serial.print(hdr.num_classes);
  Serial.println(" classes          ║");

  Serial.print("║ Sampling: ");
  Serial.print(hdr.sample_rate_hz);
  Serial.print("Hz × ");
  Serial.print(hdr.window_size);
  Serial.println(" samples        ║");

  Serial.println("╠══════════════════════════════════════╣");

  for (int i = 0; i < hdr.num_classes; i++) {
    Serial.print("║  ");
    if (i == best) Serial.print("▶ "); else Serial.print("  ");

    char buf[12];
    snprintf(buf, sizeof(buf), "%-10s", labels[i]);
    Serial.print(buf);

    // Bar graph
    int bar = (int)(scores[i] * 20);
    for (int b = 0; b < 20; b++) Serial.print(b < bar ? "█" : "░");

    Serial.print(" ");
    Serial.print(scores[i] * 100, 1);
    Serial.println("%║");
  }

  Serial.println("╚══════════════════════════════════════╝");
  Serial.println();
}

// ── Main ──
bool moduleReady = false;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000);
  delay(500);

  Serial.println();
  Serial.println("========================================");
  Serial.println(" Universal TinyML Inference Engine v1.0");
  Serial.println("========================================");
  Serial.println();

  // Check EEPROM
  Serial.print("Scanning for module EEPROM... ");
  Wire.beginTransmission(EEPROM_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println("NOT FOUND");
    Serial.println("Waiting for module...");
    return;
  }
  Serial.println("FOUND at 0x50");

  // Parse descriptor
  Serial.print("Reading descriptor... ");
  if (!parseDescriptor()) {
    Serial.println("FAILED");
    return;
  }
  Serial.println("OK");

  // Print what we found
  Serial.println();
  Serial.println("── Module Descriptor ──");
  Serial.print("  Sensor type:  ");
  switch (hdr.sensor_type) {
    case 0x01: Serial.println("Audio"); break;
    case 0x02: Serial.println("Accelerometer"); break;
    case 0x03: Serial.println("Vibration"); break;
    default:   Serial.println("Unknown"); break;
  }
  Serial.print("  Sample rate:  "); Serial.print(hdr.sample_rate_hz); Serial.println(" Hz");
  Serial.print("  Window size:  "); Serial.println(hdr.window_size);
  Serial.print("  Channels:     "); Serial.println(hdr.num_channels);
  Serial.print("  Layers:       "); Serial.println(hdr.num_layers);
  Serial.print("  Classes:      "); Serial.println(hdr.num_classes);
  Serial.print("  Labels:       ");
  for (int i = 0; i < hdr.num_classes; i++) {
    Serial.print(labels[i]);
    if (i < hdr.num_classes - 1) Serial.print(", ");
  }
  Serial.println();

  Serial.println();
  Serial.println("── Layer Architecture ──");
  const char* ltypes[] = {"?", "Conv1D", "MaxPool", "Dense", "Flatten", "ReLU", "Softmax"};
  const char* acts[] = {"none", "relu", "softmax"};
  for (int i = 0; i < hdr.num_layers; i++) {
    Serial.print("  ["); Serial.print(i); Serial.print("] ");
    int lt = layers[i].layer_type;
    Serial.print(lt <= 6 ? ltypes[lt] : "?");
    if (lt == LAYER_CONV1D) {
      Serial.print(" k="); Serial.print(layers[i].kernel_size);
      Serial.print(" f="); Serial.print(layers[i].num_filters);
    } else if (lt == LAYER_MAXPOOL1D) {
      Serial.print(" k="); Serial.print(layers[i].kernel_size);
      Serial.print(" s="); Serial.print(layers[i].stride);
    } else if (lt == LAYER_DENSE) {
      Serial.print(" "); Serial.print(layers[i].input_size);
      Serial.print("→"); Serial.print(layers[i].output_size);
    }
    if (layers[i].activation > 0) {
      Serial.print(" +"); Serial.print(acts[layers[i].activation]);
    }
    if (layers[i].weight_bytes > 0) {
      Serial.print(" ("); Serial.print(layers[i].weight_bytes); Serial.print("B)");
    }
    Serial.println();
  }

  // Load weights
  Serial.println();
  if (!loadWeights()) return;

  // Init MPU6050
  Serial.print("Initializing MPU6050... ");
  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println("NOT FOUND");
    return;
  }
  initMPU();
  Serial.println("OK");

  Serial.println();
  Serial.println("════════════════════════════════════════");
  Serial.println("  Module ready. Performing inference...");
  Serial.println("════════════════════════════════════════");
  Serial.println();

  moduleReady = true;
}

void loop() {
  if (!moduleReady) {
    delay(1000);
    return;
  }

  // Collect one window of sensor data
  int total_samples = hdr.window_size * hdr.num_channels;
  float* input = (float*)malloc(total_samples * sizeof(float));
  if (!input) { Serial.println("INPUT MALLOC FAIL"); delay(1000); return; }

  int idx = 0;
  float raw[6];
  int sample_delay = 1000 / hdr.sample_rate_hz;

  for (int i = 0; i < hdr.window_size; i++) {
    readIMU(&raw[0], &raw[1], &raw[2], &raw[3], &raw[4], &raw[5]);
    for (int j = 0; j < hdr.num_channels && j < 6; j++) {
      input[idx++] = (raw[j] - norm_means[j]) / norm_stds[j];
    }
    delay(sample_delay);
  }

  // Run inference
  float scores[DESC_MAX_CLASSES];
  runInference(input, scores);
  free(input);

  // Find best class
  int best = 0;
  for (int i = 1; i < hdr.num_classes; i++) {
    if (scores[i] > scores[best]) best = i;
  }

  // Print dashboard
  if (scores[best] > 0.5f) {
    printDashboard(scores, best);
  } else {
    Serial.print("  ... sampling (");
    Serial.print(scores[best] * 100, 1);
    Serial.println("%)");
  }
}
