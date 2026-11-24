# Parkinson's Disease Multimodal AI Diagnostic System (with Web UI)

An advanced Machine Learning clinical decision-support system that diagnoses Parkinson's Disease by fusing data across 11 high-frequency kinematic smartwatch tasks and a 30-item clinical questionnaire.

This system specifically addresses and resolves the clinical **"Essential Tremor Overlap"** problem—where early-stage Parkinsonian action tremors share frequency and amplitude similarities with Essential Tremor. By utilizing a **3-Class Weighted Ensemble** of 12 specialized Random Forest classifiers, the AI reliably distinguishes between:
1. 🟢 **Healthy Controls**
2. 🔴 **Parkinson's Disease**
3. 🟡 **Other Movement Disorders** (Essential Tremor, Multiple Sclerosis, Atypical Parkinsonism)

---

## Key Features

- **Interactive Streamlit Web Dashboard (`app.py`):**
  - Modern clinical UI with white-on-dark contrast header banner and sleek styling.
  - Sidebar patient navigator across 470 clinical study subjects with ground-truth filtering and a full-width **🎲 Pick Random Patient** button.
  - Color-coded diagnostic prediction cards with confidence scores and verification match indicators.
  - **3-Way Probability Meter:** Real-time distribution across Healthy %, Parkinson's %, and Other Movement Disorders %.
  - **💡 Diagnostic Reasoning & Probability Explanation:** An automated, natural-language clinical rationale explaining *why* the AI assigned each probability percentage based on voting consensus, non-motor markers (smell loss, REM sleep dream enactment), and sensor kinematics.
  - **📊 12-Model Ensemble Voting Tab:** Interactive Plotly horizontal bar chart displaying individual model confidences and effective ensemble voting weights.
  - **📋 Non-Motor Symptoms Tab:** Full 30-item questionnaire response inspector with highlighted early prodromal markers.
  - **👤 Patient Demographics Tab:** Detailed clinical profile, age, gender, handedness, and neurologist notes.
  - **⚙️ Global Model Architecture & Ensemble Weights:** Complete reference table at the bottom of the page detailing each task's domain, base accuracy, and ensemble voting weight.

- **Command-Line Interface (`master_diagnosis.py`):**
  - Lightweight, headless terminal mode for running batch diagnostics and testing without a browser.

- **Intelligent Engine Caching (`diagnostic_engine.py`):**
  - Caches the 12 background Random Forest models in memory so switching between patients is instantaneous with zero reload lag.

---

## Architecture Overview

The system consists of **12 specialized machine learning modules** operating under a **Late Decision Fusion** weighted ensemble architecture:

| Task Name | Data Domain | Base Accuracy | Ensemble Voting Weight | Primary Biomarkers Extracted |
| :--- | :--- | :---: | :---: | :--- |
| **Questionnaire** | Clinical Non-Motor | **77.0%** | **9.9%** | Hyposmia, REM Sleep Behavior, Dysautonomia |
| **Resting Wrist** | Kinematic (Rest) | **75.0%** | **9.6%** | 4–6 Hz Rest Tremor Power, L/R Asymmetry |
| **Drink Glass** | Kinematic (Kinetic) | **70.0%** | **9.0%** | Action Tremor, Movement Jerk, Deceleration |
| **Touch Index** | Kinematic (Coordination)| **65.7%** | **8.4%** | Target Acquisition Stability, Rapid Deceleration |
| **Entrainment** | Kinematic (Rhythm) | **65.0%** | **8.3%** | Frequency Shifting, Rhythmic Entrainment |
| **Relaxed Task** | Kinematic (Postural) | **65.0%** | **8.3%** | Postural Tremor Drift, Signal Entropy |
| **Touch Nose** | Kinematic (Target) | **63.0%** | **8.1%** | Intention Tremor, Terminal Deceleration |
| **Point Finger** | Kinematic (Postural) | **60.0%** | **7.7%** | Sustained Postural Instability |
| **Lift Hold** | Kinematic (Fatigue) | **60.0%** | **7.7%** | Tremor Amplification Over Time |
| **Stretch Hold** | Kinematic (Postural) | **60.0%** | **7.7%** | Gravity Resistance & Fatigue Ratio |
| **Hold Weight** | Kinematic (Loaded) | **60.0%** | **7.7%** | Loaded Kinetic Oscillation |
| **Cross Arms** | Kinematic (Postural) | **60.0%** | **7.7%** | Bilateral Tremor Asymmetry |

---

## Setup & Installation

### 1. Requirements
Ensure you have Python 3.9+ installed. Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

*(Packages included in `requirements.txt`: `numpy`, `pandas`, `scikit-learn`, `scipy`, `streamlit`, `plotly`)*

### 2. Dataset Setup
This project processes the **PADS (Parkinson's Disease Smartwatch Dataset)**:
1. Extract the dataset into the project root directory.
2. Confirm the directory structure matches:
```text
ParkinsonsAIProjectWithUI/
├── app.py
├── diagnostic_engine.py
├── requirements.txt
├── README.md
├── Algorithms/
│   └── Main Algorithms/
│       ├── master_diagnosis.py
│       ├── questionnaire_weightage.py
│       ├── resting_wrist_weightage.py
│       └── ... (remaining task algorithms)
└── pads-parkinsons-disease-smartwatch-dataset-1.0.0/
    ├── movement/
    ├── patients/
    └── questionnaire/
```

---

## How to Run

### Method 1: Interactive Web Dashboard (Recommended)
Open your terminal and run:
```bash
cd ~/Desktop/ParkinsonsAIProjectWithUI
python3 -m streamlit run app.py
```
*Your browser will automatically open to `http://localhost:8501`.*

### Method 2: Command-Line Terminal Interface
If you want to run a quick test from the command line:
```bash
cd ~/Desktop/ParkinsonsAIProjectWithUI
python3 "Algorithms/Main Algorithms/master_diagnosis.py" --id 28
```
*(Or omit `--id` for an interactive terminal prompt)*
