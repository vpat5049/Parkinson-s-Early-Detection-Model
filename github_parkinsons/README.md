# Parkinson's Disease Ensemble Diagnostic AI

An advanced Machine Learning ensemble model that diagnoses Parkinson's Disease by cross-referencing 11 kinematic smartwatch tasks and a 30-item clinical questionnaire. 

This project aims to solve the "Essential Tremor Overlap" problem—where early-stage Parkinson's action tremors look mathematically identical to Essential Tremors in raw smartwatch data. By fusing data across multiple modalities using a weighted ensemble of Random Forest Classifiers, the AI successfully differentiates between Healthy patients, Essential Tremor patients, and Parkinson's patients.

## Architecture
The system consists of **12 independent algorithms** that each specialize in extracting features from a specific task (e.g., drinking from a glass, resting the wrist, lifting a weight, etc.). 

1. **`master_diagnosis.py`**: The central brain. It imports all 12 modules, queries their individual Random Forest models for a specific patient, and calculates a mathematically weighted average to output a final diagnosis.
2. **`Algorithms/*_weightage.py`**: The 12 independent scripts. Each processes raw JSON sensor data (Accelerometer & Gyroscope at 100Hz), extracts kinematic features (Tremor Amplitude, Tremor Frequency 4-10Hz, Jerk, and Left/Right Asymmetry), and trains a Random Forest.

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python 3 installed. Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
This project is built to process the **PADS (Parkinson's Disease Smartwatch Dataset)**. Because of its large size, the raw data is not included in this repository.
1. Download the dataset (e.g., from PhysioNet).
2. Extract it into the root directory of this project.
3. Make sure the folder is named exactly: `pads-parkinsons-disease-smartwatch-dataset-1.0.0`

Your folder structure should look like this:
```
ParkinsonsAIProject/
│
├── Algorithms/
│   ├── master_diagnosis.py
│   ├── drinkglas_weightage.py
│   └── ... (other 11 algorithms)
│
├── pads-parkinsons-disease-smartwatch-dataset-1.0.0/
│   ├── movement/
│   ├── patients/
│   └── questionnaire/
│
├── requirements.txt
└── README.md
```

## How to Run

To run the master diagnostic tool, open your terminal and execute:
```bash
python3 Algorithms/master_diagnosis.py
```
*(On Windows, use `python` instead of `python3`)*

**What happens next?**
1. The script will quietly boot up and train all 12 Random Forests in the background (this takes ~30 seconds).
2. It will prompt you to enter a **Patient ID** (e.g., `21`, `94`, `470`).
3. It will output a detailed diagnosis report, showing exactly how each of the 12 models voted, their confidence levels, and the final weighted ensemble prediction.
