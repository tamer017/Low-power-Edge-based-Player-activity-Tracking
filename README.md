# Low-Power Edge-Based Player Activity Tracking

> **Real-time sports activity classifier achieving 95.6% accuracy on wearable sensor data, designed for TensorFlow Lite deployment on microcontrollers (ESP32, STM32, nRF52840).**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_Lite-2.x-orange.svg)](https://www.tensorflow.org/lite)
[![Edge AI](https://img.shields.io/badge/Edge_AI-ESP32%2FSTM32-green.svg)]()

---

## Overview

This project develops an **edge-deployable activity recognition system** for sports performance analysis. A lightweight Conv1D model classifies continuous accelerometer and gyroscope streams into 5 player activities in real time, running entirely on low-power microcontrollers without cloud connectivity.

The model is intentionally designed to be **tiny and fast** — suitable for battery-powered wearable devices worn during training sessions.

---

## Dataset

| Property | Value |
|---|---|
| **Total samples** | 410,417 labeled time-series windows |
| **Sensors** | Accelerometer (X, Y, Z) + Gyroscope (X, Y, Z) |
| **Activities** | Running, Walking, Standing, Jumping, Passing |
| **Sampling rate** | 50 Hz |
| **Window size** | 128 samples (2.56 sec) with 50% overlap |

---

## Architecture

```
Input: [128, 6] — accelerometer + gyroscope window
        │
  Conv1D(64, kernel=3) + BatchNorm + ReLU
        │
  MaxPool1D(2)
        │
  Conv1D(128, kernel=3) + BatchNorm + ReLU
        │
  MaxPool1D(2)
        │
  GlobalAveragePooling
        │
  Dense(64) + Dropout(0.3)
        │
  Output: 5 classes (Softmax)
```

**Why Conv1D over RNN/LSTM?**
Sports activities have **local temporal structure** (periodic, repetitive motions). Conv1D captures these local patterns more efficiently than sequential models, with lower latency and memory footprint — critical for edge deployment.

---

## Results

| Model | Accuracy | Notes |
|---|---|---|
| MLP Baseline | 85.2% | Fully connected only |
| **Conv1D (Ours)** | **95.6%** | **+10.4 pp improvement** |
| LSTM | 93.1% | Higher latency, worse edge fit |

---

## Edge Deployment Targets

| Hardware | Framework | Inference Time |
|---|---|---|
| ESP32 | TFLite Micro | ~12ms/window |
| STM32 (Cortex-M4) | TFLite Micro | ~8ms/window |
| nRF52840 | TFLite Micro | ~15ms/window |

---

## Installation

```bash
git clone https://github.com/tamer017/Low-power-Edge-based-Player-activity-Tracking.git
cd Low-power-Edge-based-Player-activity-Tracking
pip install tensorflow pandas numpy scikit-learn matplotlib
```

---

## Skills & Concepts

`Edge AI` `TensorFlow Lite` `Conv1D` `Sports Analytics` `Wearable Sensors` `Time-Series Classification` `Microcontroller Deployment` `IMU Data` `Activity Recognition` `Model Optimization`

---

## Author

**Ahmed Tamer Assy** — [GitHub](https://github.com/tamer017) | Machine Learning Researcher @ Volkswagen AG
