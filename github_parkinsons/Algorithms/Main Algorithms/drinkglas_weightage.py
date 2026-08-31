"""
DrinkGlas Task Weightage Algorithm for Parkinson's Diagnosis
=============================================================
This script determines how much WEIGHT each left and right wrist
sensor measurement during the DRINK GLASS task carries in the
final diagnosis decision.

The DrinkGlas task was ranked #1 most important active task (18.3/100).
This script dives deeper into exactly WHICH measurements during
drinking from a glass are most useful for detecting Parkinson's.

Features extracted per wrist (20 total per wrist):
  - Standard Deviation (overall shaking): 8 features
  - FFT Tremor Band 4-10 Hz (Parkinson's rhythm): 6 features
  - Movement Phases (reach, lift, drink, return): 4 features
  - Smoothness (jerk metric): 2 features
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
FS = 100.0

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
# STEP 2: Feature extraction
# ============================================================
# For the DrinkGlas task, we extract 20 features per wrist:
#
# STANDARD DEVIATION (how shaky during drinking):
#   1.  Acc Std X           - side-to-side shaking
#   2.  Acc Std Y           - up-and-down shaking
#   3.  Acc Std Z           - forward-backward shaking
#   4.  Gyro Std X          - wrist roll instability
#   5.  Gyro Std Y          - wrist pitch instability
#   6.  Gyro Std Z          - wrist yaw instability
#   7.  Max Acc Std         - worst direction of shaking
#   8.  Max Gyro Std        - worst direction of rotation
#
# FFT TREMOR BAND (4-10 Hz):
#   9.  Acc Tremor Peak Freq    - what Hz is the tremor?
#   10. Acc Tremor Peak Amp     - how strong is the tremor?
#   11. Acc Tremor Mean Power   - average tremor energy
#   12. Gyro Tremor Peak Freq   - rotational tremor frequency
#   13. Gyro Tremor Peak Amp    - rotational tremor strength
#   14. Gyro Tremor Mean Power  - average rotational tremor energy
#
# MOVEMENT PHASES (splitting recording into 4 quarters):
#   15. Q1 Acc Std (Reaching)   - shaking while reaching for glass
#   16. Q2 Acc Std (Lifting)    - shaking while lifting glass
#   17. Q3 Acc Std (Drinking)   - shaking while drinking
#   18. Q4 Acc Std (Returning)  - shaking while putting glass back
#
# SMOOTHNESS (jerk = how jerky/unsmooth the movement is):
#   19. Mean Jerk (Acc)     - average jerkiness of hand movement
#   20. Mean Jerk (Gyro)    - average jerkiness of wrist rotation
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Extracting DrinkGlas features...")
print("=" * 70)

def extract_drinkglas_features(data, fs=100.0):
    """Extract 20 features from one wrist's DrinkGlas recording."""
    acc = data[:, 1:4]
    gyro = data[:, 4:7]
    n = len(data)
    features = []

    # --- STANDARD DEVIATION (8 features) ---
    acc_std = np.std(acc, axis=0)
    gyro_std = np.std(gyro, axis=0)
    features.extend(acc_std)           # 1-3
    features.extend(gyro_std)          # 4-6
    features.append(acc_std.max())     # 7
    features.append(gyro_std.max())    # 8

    # --- FFT TREMOR BAND - Accelerometer (3 features) ---
    freqs = np.fft.fftfreq(n, 1 / fs)
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    tremor_mask = (freqs_pos >= 4.0) & (freqs_pos <= 10.0)

    best_acc_axis = np.argmax(acc_std)
    fft_acc = np.abs(np.fft.fft(acc[:, best_acc_axis]))
    fft_acc_pos = fft_acc[pos_mask]

    if tremor_mask.any():
        t_fft = fft_acc_pos[tremor_mask]
        t_freqs = freqs_pos[tremor_mask]
        pk = np.argmax(t_fft)
        features.append(t_freqs[pk])     # 9
        features.append(t_fft[pk])        # 10
        features.append(np.mean(t_fft))   # 11
    else:
        features.extend([0.0, 0.0, 0.0])

    # --- FFT TREMOR BAND - Gyroscope (3 features) ---
    best_gyro_axis = np.argmax(gyro_std)
    fft_gyro = np.abs(np.fft.fft(gyro[:, best_gyro_axis]))
    fft_gyro_pos = fft_gyro[pos_mask]

    if tremor_mask.any():
        t_fft_g = fft_gyro_pos[tremor_mask]
        t_freqs_g = freqs_pos[tremor_mask]
        pk_g = np.argmax(t_fft_g)
        features.append(t_freqs_g[pk_g])     # 12
        features.append(t_fft_g[pk_g])        # 13
        features.append(np.mean(t_fft_g))     # 14
    else:
        features.extend([0.0, 0.0, 0.0])

    # --- MOVEMENT PHASES (4 features) ---
    # Split the recording into 4 equal quarters
    quarter = n // 4
    if quarter > 10:
        q1_std = np.std(acc[:quarter], axis=0).max()           # Reaching
        q2_std = np.std(acc[quarter:2*quarter], axis=0).max()   # Lifting
        q3_std = np.std(acc[2*quarter:3*quarter], axis=0).max() # Drinking
        q4_std = np.std(acc[3*quarter:], axis=0).max()          # Returning
    else:
        q1_std = q2_std = q3_std = q4_std = acc_std.max()

    features.append(q1_std)   # 15
    features.append(q2_std)   # 16
    features.append(q3_std)   # 17
    features.append(q4_std)   # 18

    # --- SMOOTHNESS / JERK (2 features) ---
    # Jerk = rate of change of acceleration (derivative)
    # Higher jerk = jerkier, less smooth movement
    dt = 1.0 / fs
    acc_magnitude = np.sqrt(np.sum(acc**2, axis=1))
    gyro_magnitude = np.sqrt(np.sum(gyro**2, axis=1))

    if len(acc_magnitude) > 1:
        acc_jerk = np.diff(acc_magnitude) / dt
        gyro_jerk = np.diff(gyro_magnitude) / dt
        features.append(np.mean(np.abs(acc_jerk)))    # 19
        features.append(np.mean(np.abs(gyro_jerk)))   # 20
    else:
        features.extend([0.0, 0.0])

    return features

N_FEAT = 20

def get_feat_names(side):
    p = f"{side}"
    return [
        f"{p} Acc Std X (side-to-side shake)",
        f"{p} Acc Std Y (up-down shake)",
        f"{p} Acc Std Z (forward-back shake)",
        f"{p} Gyro Std X (wrist roll)",
        f"{p} Gyro Std Y (wrist pitch)",
        f"{p} Gyro Std Z (wrist yaw)",
        f"{p} Max Acc Std (worst shake axis)",
        f"{p} Max Gyro Std (worst rotation axis)",
        f"{p} Acc Tremor Peak Freq (Hz)",
        f"{p} Acc Tremor Peak Amp",
        f"{p} Acc Tremor Mean Power",
        f"{p} Gyro Tremor Peak Freq (Hz)",
        f"{p} Gyro Tremor Peak Amp",
        f"{p} Gyro Tremor Mean Power",
        f"{p} Q1 Reaching Phase Shake",
        f"{p} Q2 Lifting Phase Shake",
        f"{p} Q3 Drinking Phase Shake",
        f"{p} Q4 Returning Phase Shake",
        f"{p} Movement Jerk (Acc)",
        f"{p} Movement Jerk (Gyro)",
    ]

# ============================================================
# STEP 3: Extract for all patients
# ============================================================
all_left = []
all_right = []
all_asym = []

for i, pid in enumerate(patient_ids):
    if (i + 1) % 50 == 0:
        print(f"  Processing patient {i + 1}/{len(patient_ids)}...")

    # Left wrist
    lpath = os.path.join(TIMESERIES_DIR, f'{pid:03d}_DrinkGlas_LeftWrist.txt')
    if os.path.exists(lpath):
        try:
            ldata = np.loadtxt(lpath, delimiter=',')
            lf = extract_drinkglas_features(ldata, FS)
        except:
            lf = [0.0] * N_FEAT
    else:
        lf = [0.0] * N_FEAT

    # Right wrist
    rpath = os.path.join(TIMESERIES_DIR, f'{pid:03d}_DrinkGlas_RightWrist.txt')
    if os.path.exists(rpath):
        try:
            rdata = np.loadtxt(rpath, delimiter=',')
            rf = extract_drinkglas_features(rdata, FS)
        except:
            rf = [0.0] * N_FEAT
    else:
        rf = [0.0] * N_FEAT

    all_left.append(lf)
    all_right.append(rf)

    # Asymmetry features
    asym = [
        abs(lf[6] - rf[6]),     # Max Acc Std asymmetry
        abs(lf[7] - rf[7]),     # Max Gyro Std asymmetry
        abs(lf[9] - rf[9]),     # Acc Tremor Peak Amp asymmetry
        abs(lf[12] - rf[12]),   # Gyro Tremor Peak Amp asymmetry
        abs(lf[14] - rf[14]),   # Q1 Reaching asymmetry
        abs(lf[15] - rf[15]),   # Q2 Lifting asymmetry
        abs(lf[16] - rf[16]),   # Q3 Drinking asymmetry
        abs(lf[17] - rf[17]),   # Q4 Returning asymmetry
        abs(lf[18] - rf[18]),   # Acc Jerk asymmetry
        abs(lf[19] - rf[19]),   # Gyro Jerk asymmetry
    ]
    all_asym.append(asym)

X_left = np.array(all_left)
X_right = np.array(all_right)
X_asym = np.array(all_asym)
y = patient_labels

X = np.hstack([X_left, X_right, X_asym])

feature_names = get_feat_names("Left") + get_feat_names("Right")
feature_names += [
    "ASYMMETRY Max Acc Std",
    "ASYMMETRY Max Gyro Std",
    "ASYMMETRY Acc Tremor Peak Amp",
    "ASYMMETRY Gyro Tremor Peak Amp",
    "ASYMMETRY Q1 Reaching Phase",
    "ASYMMETRY Q2 Lifting Phase",
    "ASYMMETRY Q3 Drinking Phase",
    "ASYMMETRY Q4 Returning Phase",
    "ASYMMETRY Movement Jerk (Acc)",
    "ASYMMETRY Movement Jerk (Gyro)",
]

print(f"\n  Left wrist features: {X_left.shape[1]}")
print(f"  Right wrist features: {X_right.shape[1]}")
print(f"  Asymmetry features: {X_asym.shape[1]}")
print(f"  Total features: {X.shape[1]}")

# ============================================================
# STEP 4: Train and evaluate
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Training Random Forest (DrinkGlas only)...")
print("=" * 70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

clf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
accuracy = accuracy_score(y, y_pred)

print(f"\n  Accuracy (DrinkGlas task only): {accuracy * 100:.1f}%")
print(f"\n  Classification Report:")
print(classification_report(y, y_pred, target_names=["Healthy", "Parkinson's", "Other Movement Disorders"]))

cm = confusion_matrix(y, y_pred)
print(f"  Confusion Matrix:")
print(f"                        Predicted NOT PD    Predicted PD")
print(f"    Actual NOT PD           {cm[0][0]:>5}              {cm[0][1]:>5}")
print(f"    Actual PD               {cm[1][0]:>5}              {cm[1][1]:>5}")

# ============================================================
# STEP 5: Feature weightage (out of 100)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Feature Weightage (out of 100 points)")
print("=" * 70)

clf.fit(X_scaled, y)
importances = clf.feature_importances_

total_imp = importances.sum()
points = np.round((importances / total_imp) * 100, 1)
rounding_diff = 100.0 - points.sum()
points[np.argmax(points)] += rounding_diff

sorted_idx = np.argsort(points)[::-1]

print(f"\n  DRINKGLAS SCORING CARD (out of 100)")
print(f"  {'Rank':<5} {'Feature':<50} {'Points':>8} {'Source':>12}")
print("  " + "-" * 77)

for rank in range(len(feature_names)):
    idx = sorted_idx[rank]
    name = feature_names[idx]
    if 'ASYMMETRY' in name:
        source = "ASYMMETRY"
    elif 'Left' in name:
        source = "LEFT"
    else:
        source = "RIGHT"
    print(f"  {rank+1:<5} {name:<50} {points[idx]:>7.1f} pts {source:>10}")

# ============================================================
# STEP 6: Category breakdowns
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Weightage Breakdown by Category")
print("=" * 70)

# --- BY WRIST ---
left_total = sum(points[i] for i in range(N_FEAT))
right_total = sum(points[i] for i in range(N_FEAT, 2 * N_FEAT))
asym_total = sum(points[i] for i in range(2 * N_FEAT, len(points)))

print(f"\n  BY WRIST:")
print(f"    Left Wrist:     {left_total:>5.1f} / 100 pts  {'█' * int(left_total / 2)}")
print(f"    Right Wrist:    {right_total:>5.1f} / 100 pts  {'█' * int(right_total / 2)}")
print(f"    Asymmetry:      {asym_total:>5.1f} / 100 pts  {'█' * int(asym_total / 2)}")

# --- BY SENSOR ---
acc_indices = [i for i, n in enumerate(feature_names) if 'Acc' in n and 'ASYMMETRY' not in n]
gyro_indices = [i for i, n in enumerate(feature_names) if 'Gyro' in n and 'ASYMMETRY' not in n]

acc_total = sum(points[i] for i in acc_indices)
gyro_total = sum(points[i] for i in gyro_indices)

print(f"\n  BY SENSOR:")
print(f"    Accelerometer:  {acc_total:>5.1f} / 100 pts  {'█' * int(acc_total / 2)}")
print(f"    Gyroscope:      {gyro_total:>5.1f} / 100 pts  {'█' * int(gyro_total / 2)}")
print(f"    Asymmetry:      {asym_total:>5.1f} / 100 pts  {'█' * int(asym_total / 2)}")

# --- BY MEASUREMENT TYPE ---
std_indices = [i for i, n in enumerate(feature_names)
               if ('Std X' in n or 'Std Y' in n or 'Std Z' in n
                   or 'Max Acc' in n or 'Max Gyro' in n)
               and 'ASYMMETRY' not in n and 'Phase' not in n]
fft_indices = [i for i, n in enumerate(feature_names)
               if 'Tremor' in n and 'ASYMMETRY' not in n]
phase_indices = [i for i, n in enumerate(feature_names)
                 if ('Q1' in n or 'Q2' in n or 'Q3' in n or 'Q4' in n)
                 and 'ASYMMETRY' not in n]
jerk_indices = [i for i, n in enumerate(feature_names)
                if 'Jerk' in n and 'ASYMMETRY' not in n]

std_total = sum(points[i] for i in std_indices)
fft_total = sum(points[i] for i in fft_indices)
phase_total = sum(points[i] for i in phase_indices)
jerk_total = sum(points[i] for i in jerk_indices)

print(f"\n  BY MEASUREMENT TYPE:")
print(f"    Std Deviation (overall shake):    {std_total:>5.1f} / 100 pts  {'█' * int(std_total / 2)}")
print(f"    FFT Tremor Band (4-10 Hz):        {fft_total:>5.1f} / 100 pts  {'█' * int(fft_total / 2)}")
print(f"    Movement Phases (reach/lift/etc):  {phase_total:>5.1f} / 100 pts  {'█' * int(phase_total / 2)}")
print(f"    Jerk (movement smoothness):        {jerk_total:>5.1f} / 100 pts  {'█' * int(jerk_total / 2)}")
print(f"    Asymmetry (L vs R difference):     {asym_total:>5.1f} / 100 pts  {'█' * int(asym_total / 2)}")

# --- BY MOVEMENT PHASE ---
print(f"\n  BY MOVEMENT PHASE (which part of drinking matters most):")
phase_names_list = ['Q1 Reaching', 'Q2 Lifting', 'Q3 Drinking', 'Q4 Returning']
for pi, pname in enumerate(phase_names_list):
    # Left + Right + Asymmetry for this phase
    phase_feat_idx = 14 + pi  # index within one wrist's features
    left_pts = points[phase_feat_idx]
    right_pts = points[N_FEAT + phase_feat_idx]
    asym_idx = 2 * N_FEAT + 4 + pi  # asymmetry index
    asym_pts = points[asym_idx]
    total_phase = left_pts + right_pts + asym_pts
    print(f"    {pname:<15} {total_phase:>5.1f} / 100 pts  {'█' * int(total_phase / 2)}")

# ============================================================
# STEP 7: Diagnosing Patient(s)
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Diagnosing Patient(s)")
print("=" * 70)

import argparse
parser = argparse.ArgumentParser(description="DrinkGlas Weightage & Diagnosis")
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
        
        # Key stats
        l_max_shake = X_left[idx][6]
        r_max_shake = X_right[idx][6]
        l_jerk = X_left[idx][18]
        r_jerk = X_right[idx][18]
        l_tremor = X_left[idx][9]
        r_tremor = X_right[idx][9]

        print(f"  Left Wrist:  Shake = {l_max_shake:.4f} g | Tremor Amp = {l_tremor:.2f} | Jerk = {l_jerk:.2f}")
        print(f"  Right Wrist: Shake = {r_max_shake:.4f} g | Tremor Amp = {r_tremor:.2f} | Jerk = {r_jerk:.2f}")
        print(f"  Asymmetry:   {abs(l_max_shake - r_max_shake):.4f} g")
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
