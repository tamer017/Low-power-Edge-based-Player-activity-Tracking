# Low-Power Edge-Based Player Activity Tracking

> A sports analytics system for real-time player activity classification using a **1D Convolutional Neural Network** deployed on low-power edge hardware.

[![Framework](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange?style=flat-square)](https://www.tensorflow.org/)
[![Accuracy](https://img.shields.io/badge/Best%20Accuracy-95.6%25-brightgreen?style=flat-square)]()
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-blue?style=flat-square)](https://colab.research.google.com/)

---

## Overview

This project builds a machine learning pipeline to classify **5 distinct player activities** — running, walking, standing, jumping, and passing — from body-mounted sensor data. Targeting low-power edge deployment in sports environments, the system uses a **1D Convolutional Neural Network (Conv1D)** trained on a large-scale sensor dataset comprising over **410,000 labeled samples**.

The model achieves a best accuracy of **95.6%** on the held-out test set, outperforming a baseline accuracy of 85.2% by **10.4 percentage points**.

---

## Dataset

| Activity | Samples |
|---|---|
| Running | 98,916 |
| Walking | 85,616 |
| Standing | 82,777 |
| Jumping | 68,505 |
| Passing | 74,603 |
| **Total** | **410,417** |

- **Format:** CSV files, one per activity class
- **Features:** Multi-axis accelerometer / gyroscope sensor readings
- **Preprocessing:** MinMax scaling, label encoding, train/test split

---

## Model Architecture

```
[Input: Sensor Frame]
        |
        v
  [Conv1D (filters=64, kernel=3)] + BatchNorm + ReLU
        |
        v
  [MaxPooling1D (pool=2)]
        |
        v
  [Conv1D (filters=128, kernel=3)] + BatchNorm + ReLU
        |
        v
  [MaxPooling1D (pool=2)]
        |
        v
  [Flatten]
        |
        v
  [Dense (256)] + Dropout(0.5)
        |
        v
  [Dense (5)] -> Softmax
        |
        v
  [Activity Class Output]
```

---

## Technical Highlights

### 1D Convolutional Feature Extraction
Conv1D layers treat sensor time-series windows as 1D signals, learning local temporal patterns (e.g., the rhythmic periodicity of running vs. the irregular spikes of jumping) at multiple filter scales.

### Training Configuration
| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Dropout rate | 0.5 |
| Epochs (max) | 50 |

### Results
| Metric | Baseline | This Model |
|---|---|---|
| Test Accuracy | 85.2% | **95.6%** |
| Improvement | — | **+10.4pp** |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/tamer017/Low-power-Edge-based-Player-activity-Tracking.git
cd Low-power-Edge-based-Player-activity-Tracking

# Install dependencies
pip install tensorflow pandas numpy matplotlib scikit-learn

# Open the notebook
jupyter notebook Low_power_Edge_based_Player_activity_Tracking.ipynb
```

> **Note:** The dataset CSV files must be placed in the working directory. Update file paths in the notebook if needed.

---

## Skills Demonstrated

- **Deep Learning:** TensorFlow 2.x, Keras Sequential API, Conv1D, MaxPooling1D, Dropout
- **Data Engineering:** Pandas, NumPy, multi-file CSV loading, feature scaling
- **Sports Analytics:** Wearable sensor data classification, activity recognition systems
- **Model Evaluation:** Accuracy, F1-score, confusion matrix visualization
- **Tooling:** Google Colab, Matplotlib, Scikit-learn

---

## Future Work

- [ ] TensorFlow Lite conversion for deployment on microcontrollers (e.g., Arduino Nano 33 BLE Sense)
- [ ] Sliding window segmentation for real-time inference
- [ ] Federated learning across multiple players’ devices
- [ ] Integration with wearable IoT hardware (Raspberry Pi Zero, STM32)
