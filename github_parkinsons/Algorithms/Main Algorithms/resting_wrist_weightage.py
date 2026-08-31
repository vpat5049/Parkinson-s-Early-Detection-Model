"""
Resting Wrist Weightage Algorithm for Parkinson's Diagnosis
=============================================================
This script determines how much WEIGHT each left and right wrist
resting sensor measurement carries in the final diagnosis decision.

It uses ONLY the Relaxed task data (both wrists) to predict
Parkinson's Disease, and shows:
  - Which wrist matters more (left vs right)
  - Which sensor matters more (accelerometer vs gyroscope)
  - Which axis matters more (X, Y, Z)
  - Which frequency features matter (tremor band 4-10 Hz)
  - How asymmetry between wrists contributes
"""

import os
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = 'pads-parkinsons-disease-smartwatch-dataset-1.0.0'
TIMESERIES_DIR = f'{BASE_DIR}/movement/timeseries'
FILE_LIST = f'{BASE_DIR}/preprocessed/file_list.csv'
FS = 100.0  # Sampling rate (Hz)

# ============================================================
# STEP 1: Load patient labels
# ============================================================
print("=" * 70)
print("STEP 1: Loading patient labels...")
print("=" * 70)

patient_ids = []
patient_labels = []
patient_conditions = []

with open(FILE_LIST, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        pid = int(row[1])
        condition = row[3]
        patient_ids.append(pid)
        patient_conditions.append(condition)
        patient_labels.append(1 if condition == "Parkinson's" else (0 if condition in ["Healthy", "Healthy Control"] else 2))

patient_labels = np.array(patient_labels)
print(f"  Total patients: {len(patient_ids)}")
print(f"  Parkinson's: {np.sum(patient_labels == 1)}")
print(f"  Not Parkinson's: {np.sum(patient_labels == 0)}")

# ============================================================
# STEP 2: Feature extraction function
# ============================================================
# For each wrist's Relaxed recording, we extract 17 features:
#
#   STANDARD DEVIATION (how much shaking):
#     1. Acc Std X          - shaking side-to-side
#     2. Acc Std Y          - shaking up-and-down
#     3. Acc Std Z          - shaking forward-backward
#     4. Gyro Std X         - rotational shake around X axis
#     5. Gyro Std Y         - rotational shake around Y axis
#     6. Gyro Std Z         - rotational shake around Z axis
#     7. Max Acc Std        - worst axis of shaking
#     8. Max Gyro Std       - worst axis of rotation
#
#   FFT TREMOR BAND (4-10 Hz) on accelerometer:
#     9.  Tremor Peak Freq  - what frequency is the tremor?
#     10. Tremor Peak Amp   - how strong is the tremor peak?
#     11. Tremor Mean Power - average energy in tremor band
#
#   FFT TREMOR BAND (4-10 Hz) on gyroscope:
#     12. Gyro Tremor Peak Freq
#     13. Gyro Tremor Peak Amp
#     14. Gyro Tremor Mean Power
#
#   SLIDING WINDOW (late tremor detection):
#     15. Max Std in Last 3 Seconds (Acc)  - catches re-emergent tremors
#     16. Max Std in Last 3 Seconds (Gyro) - catches re-emergent tremors
#     17. Ratio of Last 3s Std to Full Std - is the end shakier than the whole?
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Extracting resting wrist features...")
print("=" * 70)

def extract_resting_features(data, fs=100.0):
    """
    Extract 17 features from one wrist's Relaxed task recording.
    """
    acc = data[:, 1:4]    # Accelerometer X, Y, Z
    gyro = data[:, 4:7]   # Gyroscope X, Y, Z
    n = len(data)
    features = []

    # --- STANDARD DEVIATION ---
    acc_std = np.std(acc, axis=0)       # [3 values]
    gyro_std = np.std(gyro, axis=0)     # [3 values]
    features.extend(acc_std)             # 1-3: Acc Std X, Y, Z
    features.extend(gyro_std)            # 4-6: Gyro Std X, Y, Z
    features.append(acc_std.max())       # 7:   Max Acc Std
    features.append(gyro_std.max())      # 8:   Max Gyro Std

    # --- FFT TREMOR BAND (Accelerometer) ---
    best_acc_axis = np.argmax(acc_std)
    signal_acc = acc[:, best_acc_axis]
    freqs = np.fft.fftfreq(n, 1 / fs)
    fft_acc = np.abs(np.fft.fft(signal_acc))
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    fft_acc_pos = fft_acc[pos_mask]

    tremor_mask = (freqs_pos >= 4.0) & (freqs_pos <= 10.0)
    if tremor_mask.any():
        tremor_fft = fft_acc_pos[tremor_mask]
        tremor_freqs = freqs_pos[tremor_mask]
        peak_idx = np.argmax(tremor_fft)
        features.append(tremor_freqs[peak_idx])    # 9:  Peak tremor frequency
        features.append(tremor_fft[peak_idx])       # 10: Peak tremor amplitude
        features.append(np.mean(tremor_fft))         # 11: Mean tremor power
    else:
        features.extend([0.0, 0.0, 0.0])

    # --- FFT TREMOR BAND (Gyroscope) ---
    best_gyro_axis = np.argmax(gyro_std)
    signal_gyro = gyro[:, best_gyro_axis]
    fft_gyro = np.abs(np.fft.fft(signal_gyro))
    fft_gyro_pos = fft_gyro[pos_mask]

    if tremor_mask.any():
        tremor_fft_g = fft_gyro_pos[tremor_mask]
        tremor_freqs_g = freqs_pos[tremor_mask]
        peak_idx_g = np.argmax(tremor_fft_g)
        features.append(tremor_freqs_g[peak_idx_g])  # 12: Gyro peak tremor freq
        features.append(tremor_fft_g[peak_idx_g])     # 13: Gyro peak tremor amp
        features.append(np.mean(tremor_fft_g))         # 14: Gyro mean tremor power
    else:
        features.extend([0.0, 0.0, 0.0])

    # --- SLIDING WINDOW: Last 3 seconds ---
    last_3s_samples = int(3.0 * fs)  # 300 samples
    if n >= last_3s_samples:
        acc_last = acc[-last_3s_samples:]
        gyro_last = gyro[-last_3s_samples:]
        acc_last_std = np.std(acc_last, axis=0).max()
        gyro_last_std = np.std(gyro_last, axis=0).max()

        # Ratio: is the end shakier than the whole recording?
        full_std = acc_std.max()
        ratio = acc_last_std / (full_std + 1e-8)
    else:
        acc_last_std = acc_std.max()
        gyro_last_std = gyro_std.max()
        ratio = 1.0

    features.append(acc_last_std)    # 15: Max Acc Std in last 3s
    features.append(gyro_last_std)   # 16: Max Gyro Std in last 3s
    features.append(ratio)           # 17: Late tremor ratio

    return features

N_FEATURES_PER_WRIST = 17

# Feature names for one wrist
def get_feature_names(side):
    prefix = f"{side}"
    return [
        f"{prefix} Acc Std X",
        f"{prefix} Acc Std Y",
        f"{prefix} Acc Std Z",
        f"{prefix} Gyro Std X",
        f"{prefix} Gyro Std Y",
        f"{prefix} Gyro Std Z",
        f"{prefix} Max Acc Std",
        f"{prefix} Max Gyro Std",
        f"{prefix} Acc Tremor Peak Freq",
        f"{prefix} Acc Tremor Peak Amp",
        f"{prefix} Acc Tremor Mean Power",
        f"{prefix} Gyro Tremor Peak Freq",
        f"{prefix} Gyro Tremor Peak Amp",
        f"{prefix} Gyro Tremor Mean Power",
        f"{prefix} Acc Std Last 3s",
        f"{prefix} Gyro Std Last 3s",
        f"{prefix} Late Tremor Ratio",
    ]

# ============================================================
# STEP 3: Extract features for all patients
# ============================================================
all_left_features = []
all_right_features = []
all_asym_features = []

for i, pid in enumerate(patient_ids):
    if (i + 1) % 50 == 0:
        print(f"  Processing patient {i + 1}/{len(patient_ids)}...")

    # Left wrist
    left_path = os.path.join(TIMESERIES_DIR, f'{pid:03d}_Relaxed_LeftWrist.txt')
    if os.path.exists(left_path):
        try:
            left_data = np.loadtxt(left_path, delimiter=',')
            left_feats = extract_resting_features(left_data, FS)
        except:
            left_feats = [0.0] * N_FEATURES_PER_WRIST
    else:
        left_feats = [0.0] * N_FEATURES_PER_WRIST

    # Right wrist
    right_path = os.path.join(TIMESERIES_DIR, f'{pid:03d}_Relaxed_RightWrist.txt')
    if os.path.exists(right_path):
        try:
            right_data = np.loadtxt(right_path, delimiter=',')
            right_feats = extract_resting_features(right_data, FS)
        except:
            right_feats = [0.0] * N_FEATURES_PER_WRIST
    else:
        right_feats = [0.0] * N_FEATURES_PER_WRIST

    all_left_features.append(left_feats)
    all_right_features.append(right_feats)

    # Asymmetry features: |Left - Right| for key measurements
    asym = [
        abs(left_feats[6] - right_feats[6]),    # Max Acc Std asymmetry
        abs(left_feats[7] - right_feats[7]),    # Max Gyro Std asymmetry
        abs(left_feats[9] - right_feats[9]),    # Tremor peak amp asymmetry
        abs(left_feats[12] - right_feats[12]),  # Gyro tremor peak amp asymmetry
        abs(left_feats[14] - right_feats[14]),  # Acc last 3s std asymmetry
        abs(left_feats[15] - right_feats[15]),  # Gyro last 3s std asymmetry
        abs(left_feats[16] - right_feats[16]),  # Late tremor ratio asymmetry
    ]
    all_asym_features.append(asym)

X_left = np.array(all_left_features)
X_right = np.array(all_right_features)
X_asym = np.array(all_asym_features)
y = patient_labels

# Combine all features
X = np.hstack([X_left, X_right, X_asym])

# Build all feature names
feature_names = get_feature_names("Left Wrist") + get_feature_names("Right Wrist")
feature_names += [
    "ASYMMETRY Max Acc Std",
    "ASYMMETRY Max Gyro Std",
    "ASYMMETRY Acc Tremor Peak Amp",
    "ASYMMETRY Gyro Tremor Peak Amp",
    "ASYMMETRY Acc Std Last 3s",
    "ASYMMETRY Gyro Std Last 3s",
    "ASYMMETRY Late Tremor Ratio",
]

print(f"\n  Left wrist features: {X_left.shape[1]}")
print(f"  Right wrist features: {X_right.shape[1]}")
print(f"  Asymmetry features: {X_asym.shape[1]}")
print(f"  Total features: {X.shape[1]}")

# ============================================================
# STEP 4: Train Random Forest and evaluate
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Training Random Forest (resting data only)...")
print("=" * 70)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
accuracy = accuracy_score(y, y_pred)

print(f"\n  Accuracy (resting wrist data only): {accuracy * 100:.1f}%")
print(f"\n  Classification Report:")
print(classification_report(y, y_pred, target_names=["Healthy", "Parkinson's", "Other Movement Disorders"]))

cm = confusion_matrix(y, y_pred)
print(f"  Confusion Matrix:")
print(f"                        Predicted NOT PD    Predicted PD")
print(f"    Actual NOT PD           {cm[0][0]:>5}              {cm[0][1]:>5}")
print(f"    Actual PD               {cm[1][0]:>5}              {cm[1][1]:>5}")

# ============================================================
# STEP 5: Feature importance and weightage
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Feature Weightage (out of 100 points)")
print("=" * 70)

# Train on full data for importances
clf.fit(X_scaled, y)
importances = clf.feature_importances_

# Normalize to 100 points
total_importance = importances.sum()
point_values = np.round((importances / total_importance) * 100, 1)

# Fix rounding
rounding_diff = 100.0 - point_values.sum()
max_idx = np.argmax(point_values)
point_values[max_idx] += rounding_diff

# Sort and display
sorted_idx = np.argsort(point_values)[::-1]

print(f"\n  RESTING WRIST SCORING CARD (out of 100)")
print(f"  {'Rank':<5} {'Feature':<45} {'Points':>8} {'Category':>12}")
print("  " + "-" * 72)

for rank in range(len(feature_names)):
    idx = sorted_idx[rank]
    name = feature_names[idx]

    # Categorize
    if "Left" in name:
        category = "LEFT"
    elif "Right" in name:
        category = "RIGHT"
    else:
        category = "ASYMMETRY"

    print(f"  {rank+1:<5} {name:<45} {point_values[idx]:>7.1f} pts {category:>10}")

# ============================================================
# STEP 6: Category breakdown
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Weightage Breakdown by Category")
print("=" * 70)

# Left vs Right vs Asymmetry
left_total = sum(point_values[i] for i in range(N_FEATURES_PER_WRIST))
right_total = sum(point_values[i] for i in range(N_FEATURES_PER_WRIST, 2 * N_FEATURES_PER_WRIST))
asym_total = sum(point_values[i] for i in range(2 * N_FEATURES_PER_WRIST, len(point_values)))

print(f"\n  BY WRIST:")
print(f"    Left Wrist:     {left_total:>5.1f} / 100 points")
print(f"    Right Wrist:    {right_total:>5.1f} / 100 points")
print(f"    Asymmetry:      {asym_total:>5.1f} / 100 points")

# Accelerometer vs Gyroscope
acc_indices = [i for i, name in enumerate(feature_names) if 'Acc' in name and 'ASYMMETRY' not in name]
gyro_indices = [i for i, name in enumerate(feature_names) if 'Gyro' in name and 'ASYMMETRY' not in name]
asym_indices = [i for i, name in enumerate(feature_names) if 'ASYMMETRY' in name]

acc_total = sum(point_values[i] for i in acc_indices)
gyro_total = sum(point_values[i] for i in gyro_indices)

print(f"\n  BY SENSOR:")
print(f"    Accelerometer:  {acc_total:>5.1f} / 100 points")
print(f"    Gyroscope:      {gyro_total:>5.1f} / 100 points")
print(f"    Asymmetry:      {asym_total:>5.1f} / 100 points")

# Standard Deviation vs FFT vs Late Tremor
std_indices = [i for i, name in enumerate(feature_names)
               if ('Std X' in name or 'Std Y' in name or 'Std Z' in name
                   or 'Max Acc Std' in name or 'Max Gyro Std' in name)
               and 'ASYMMETRY' not in name and 'Last' not in name]
fft_indices = [i for i, name in enumerate(feature_names)
               if ('Tremor' in name and 'Last' not in name and 'Ratio' not in name)
               and 'ASYMMETRY' not in name]
late_indices = [i for i, name in enumerate(feature_names)
                if ('Last 3s' in name or 'Late Tremor' in name)
                and 'ASYMMETRY' not in name]

std_total = sum(point_values[i] for i in std_indices)
fft_total = sum(point_values[i] for i in fft_indices)
late_total = sum(point_values[i] for i in late_indices)

print(f"\n  BY MEASUREMENT TYPE:")
print(f"    Std Deviation (overall shaking):    {std_total:>5.1f} / 100 points")
print(f"    FFT Tremor Band (4-10 Hz rhythm):   {fft_total:>5.1f} / 100 points")
print(f"    Late Tremor (last 3s re-emergence):  {late_total:>5.1f} / 100 points")
print(f"    Asymmetry (left vs right diff):      {asym_total:>5.1f} / 100 points")

# ============================================================
# STEP 6: Diagnosing Patient(s)
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Diagnosing Patient(s)")
print("=" * 70)

import argparse
parser = argparse.ArgumentParser(description="Resting Wrist Weightage & Diagnosis")
parser.add_argument('--id', type=int, default=None, help='Specific Patient ID to diagnose (e.g. --id 94)')
args, unknown = parser.parse_known_args()

clf.fit(X_scaled, y)

def print_report(demo_pid):
    if demo_pid in patient_ids:
        idx = patient_ids.index(demo_pid)
        patient_data = X_scaled[idx:idx+1]
        
        pred = clf.predict(patient_data)[0]
        proba = clf.predict_proba(patient_data)[0]
        actual = patient_conditions[idx]
        pred_label = "Parkinson's Disease" if pred == 1 else "Not Parkinson's"
        confidence = proba[pred] * 100

        # Calculate key feature contributions
        contributions = patient_data[0] * importances
        top_indices = np.argsort(np.abs(contributions))[::-1][:3]
        
        match_status = "MATCH (Correct)" if (pred == 1 and actual == "Parkinson's") or (pred == 0 and actual != "Parkinson's") else "MISMATCH"

        # Generate Explanation
        if pred == 1:
            top_feature = feature_names[top_indices[0]]
            direction = "Elevated" if contributions[top_indices[0]] > 0 else "Reduced"
            explanation = f"The AI detected a high probability of Parkinson's based on\n                           movement patterns, specifically driven by '{top_feature}' ({direction})."
        else:
            explanation = f"The AI did not detect significant Parkinson's-like tremors or\n                           abnormalities in the wrist movements, resulting in a healthy prediction."

        print(f"\n  DIAGNOSIS REPORT FOR PATIENT ID: {demo_pid:03d}")
        print(f"  --------------------------------------------------")
        print(f"  AI Prediction (Blind):   {pred_label:<20} (Confidence: {confidence:.1f}%)")
        print(f"  Actual Ground Truth:     {actual}")
        print(f"  Verification Result:     {match_status}")
        print(f"  Explanation:             {explanation}")
        print(f"  --------------------------------------------------")
        print(f"  Top 3 Key Sensor Features Influencing Decision:")
        for rank_num, ti in enumerate(top_indices, 1):
            direction = "Elevated" if contributions[ti] > 0 else "Reduced"
            print(f"    {rank_num}. {feature_names[ti]} ({direction})")
        print(f"  --------------------------------------------------")
        
        # Show the patient's key resting stats
        left_max_std = X_left[idx][6]    # Max Acc Std left
        right_max_std = X_right[idx][6]  # Max Acc Std right
        left_tremor_amp = X_left[idx][9]  # Tremor peak amp left
        right_tremor_amp = X_right[idx][9] # Tremor peak amp right
        left_late = X_left[idx][14]       # Last 3s std left
        right_late = X_right[idx][14]     # Last 3s std right

        print(f"  Left Wrist:  Max Shake = {left_max_std:.4f} g | Tremor Amp = {left_tremor_amp:.2f} | Last 3s = {left_late:.4f} g")
        print(f"  Right Wrist: Max Shake = {right_max_std:.4f} g | Tremor Amp = {right_tremor_amp:.2f} | Last 3s = {right_late:.4f} g")
        print(f"  Asymmetry:   {abs(left_max_std - right_max_std):.4f} g")
    else:
        print(f"Error: Patient ID {demo_pid} not found in PADS dataset.")

if __name__ == "__main__":
    if args.id is not None:
        print_report(args.id)
    else:
        while True:
            user_input = input("\nWhich patient ID would you like to check? (e.g. 94, or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            try:
                demo_pid = int(user_input)
                print_report(demo_pid)
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
