"""
Questionnaire Weightage Algorithm for Parkinson's Diagnosis
============================================================
This script determines how much WEIGHT each of the 30 questionnaire
questions carries in the final diagnosis decision.

It calculates weightage using 3 methods:
  1. Statistical Weightage   - Based on how differently PD vs Healthy answer
  2. Random Forest Weightage - Based on what the AI model relies on most
  3. Combined Risk Score     - A simple scoring system you can use by hand
"""

import os
import csv
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = 'pads-parkinsons-disease-smartwatch-dataset-1.0.0'
QUESTIONNAIRE_DIR = f'{BASE_DIR}/questionnaire'
FILE_LIST = f'{BASE_DIR}/preprocessed/file_list.csv'

# ============================================================
# STEP 1: Load patient labels from file_list.csv
# ============================================================
print("=" * 70)
print("STEP 1: Loading patient labels...")
print("=" * 70)

patient_ids = []
patient_labels = []     # 1 = Parkinson's, 0 = Not Parkinson's
patient_conditions = []

with open(FILE_LIST, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        pid = int(row[1])
        condition = row[3]
        patient_ids.append(pid)
        patient_conditions.append(condition)
        patient_labels.append(1 if condition == "Parkinson's" else 0)

patient_labels = np.array(patient_labels)
print(f"  Total patients: {len(patient_ids)}")
print(f"  Parkinson's: {np.sum(patient_labels == 1)}")
print(f"  Not Parkinson's: {np.sum(patient_labels == 0)}")

# ============================================================
# STEP 2: Load questionnaire responses
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Loading questionnaire responses...")
print("=" * 70)

all_answers = []
question_texts = None

for pid in patient_ids:
    fname = f'questionnaire_response_{pid:03d}.json'
    fpath = os.path.join(QUESTIONNAIRE_DIR, fname)

    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            data = json.load(f)
        answers = [1 if item['answer'] == True else 0 for item in data['item']]
        if question_texts is None:
            question_texts = [item['text'] for item in data['item']]
        all_answers.append(answers)
    else:
        all_answers.append([0] * 30)

X = np.array(all_answers)   # Shape: (469 patients, 30 questions)
y = patient_labels           # Shape: (469,)

print(f"  Feature matrix: {X.shape}")
print(f"  (469 patients x 30 questions)")

# ============================================================
# STEP 3: METHOD 1 - Statistical Weightage
# ============================================================
# This calculates weight based on two factors:
#   (a) How much more common is a "True" answer in PD vs Healthy?
#   (b) How specific is a "True" answer to PD? (low false positive rate)
#
# Formula: Weight = (PD_rate - Healthy_rate) * Specificity_bonus
#   where Specificity_bonus = 1 / (Healthy_rate + 0.01)
#   This rewards questions where healthy people rarely say True
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: METHOD 1 - Statistical Weightage")
print("=" * 70)

pd_mask = (y == 1)
healthy_mask = (patient_labels == 0) & np.array([c == "Healthy" for c in patient_conditions])

# Calculate True rates for PD and Healthy groups
pd_rates = X[pd_mask].mean(axis=0)         # % of PD patients answering True
healthy_rates = X[healthy_mask].mean(axis=0) # % of Healthy patients answering True

# Calculate statistical weight for each question
# Gap: how much more PD patients answer True vs Healthy
gap = pd_rates - healthy_rates

# Specificity bonus: reward questions where healthy people rarely say True
# A question where 0% of healthy say True gets a much higher bonus
specificity_bonus = 1.0 / (healthy_rates + 0.01)

# Combined statistical weight (normalize to sum to 100%)
raw_stat_weight = gap * specificity_bonus
raw_stat_weight = np.maximum(raw_stat_weight, 0)  # Remove negatives
stat_weight = (raw_stat_weight / raw_stat_weight.sum()) * 100  # Normalize to %

print(f"\n  {'Rank':<5} {'Question':<65} {'PD %':>6} {'Healthy %':>10} {'Weight':>8}")
print("  " + "-" * 96)

stat_sorted = np.argsort(stat_weight)[::-1]
for rank, idx in enumerate(stat_sorted, 1):
    q_text = question_texts[idx]
    if len(q_text) > 60:
        q_text = q_text[:57] + '...'
    print(f"  {rank:<5} {q_text:<65} {pd_rates[idx]*100:>5.1f}% {healthy_rates[idx]*100:>9.1f}% {stat_weight[idx]:>7.2f}%")

# ============================================================
# STEP 4: METHOD 2 - Random Forest Weightage
# ============================================================
# The Random Forest model learns which questions are most useful
# for separating PD from non-PD patients. It assigns an
# "importance" score to each question based on how much
# each question improves prediction accuracy.
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: METHOD 2 - Random Forest AI Weightage")
print("=" * 70)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# 5-fold cross-validation to get unbiased accuracy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(clf, X, y, cv=cv)
accuracy = accuracy_score(y, y_pred)

# Train on all data to get feature importances
clf.fit(X, y)
rf_importances = clf.feature_importances_

# Normalize to percentages
rf_weight = (rf_importances / rf_importances.sum()) * 100

print(f"\n  Model Accuracy (questionnaire only): {accuracy * 100:.1f}%")
print(f"\n  {'Rank':<5} {'Question':<65} {'AI Weight':>10}")
print("  " + "-" * 82)

rf_sorted = np.argsort(rf_weight)[::-1]
for rank, idx in enumerate(rf_sorted, 1):
    q_text = question_texts[idx]
    if len(q_text) > 60:
        q_text = q_text[:57] + '...'
    print(f"  {rank:<5} {q_text:<65} {rf_weight[idx]:>9.2f}%")

# ============================================================
# STEP 5: METHOD 3 - Combined Risk Score System
# ============================================================
# This creates a simple point-based scoring system that
# combines both methods. Each question gets a point value
# based on its combined statistical + AI weight.
#
# To use: Add up the points for every "True" answer.
# If the total exceeds the threshold -> predict Parkinson's.
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: METHOD 3 - Combined Risk Score System")
print("=" * 70)

# Combine both methods (50% statistical, 50% AI)
combined_weight = (stat_weight + rf_weight) / 2.0

# Normalize so all 30 questions sum to exactly 100 points
# Each question's points = its share of the total weight, scaled to 100
total_weight = combined_weight.sum()
point_values = np.round((combined_weight / total_weight) * 100, 1)

# Fix rounding so it sums to exactly 100
rounding_diff = 100.0 - point_values.sum()
max_idx = np.argmax(point_values)
point_values[max_idx] += rounding_diff

print(f"\n  PARKINSON'S RISK SCORING CARD")
print(f"  {'Rank':<5} {'Question':<65} {'Points':>7}")
print("  " + "-" * 79)

combined_sorted = np.argsort(combined_weight)[::-1]
for rank, idx in enumerate(combined_sorted, 1):
    q_text = question_texts[idx]
    if len(q_text) > 60:
        q_text = q_text[:57] + '...'
    points_str = f"{point_values[idx]:.1f} pts"
    print(f"  {rank:<5} {q_text:<65} {points_str:>7}")

# ============================================================
# STEP 6: Find the optimal threshold
# ============================================================
# Calculate each patient's total risk score, then find the
# best cutoff point that separates PD from non-PD.
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Finding optimal risk score threshold...")
print("=" * 70)

# Calculate risk score for every patient
patient_scores = np.zeros(len(patient_ids))
for i in range(len(patient_ids)):
    # Sum up points for every question where the patient answered True
    patient_scores[i] = np.sum(X[i] * point_values)

# Try every possible threshold and find the one with best accuracy
best_threshold = 0
best_accuracy = 0
best_sensitivity = 0
best_specificity = 0

for threshold in np.arange(0, patient_scores.max(), 0.5):
    predictions = (patient_scores >= threshold).astype(int)
    acc = accuracy_score(y, predictions)

    # Sensitivity (how many PD patients are caught)
    true_positives = np.sum((predictions == 1) & (y == 1))
    total_positives = np.sum(y == 1)
    sensitivity = true_positives / total_positives if total_positives > 0 else 0

    # Specificity (how many healthy are correctly identified)
    true_negatives = np.sum((predictions == 0) & (y == 0))
    total_negatives = np.sum(y == 0)
    specificity = true_negatives / total_negatives if total_negatives > 0 else 0

    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold
        best_sensitivity = sensitivity
        best_specificity = specificity

print(f"\n  Optimal Threshold: {best_threshold:.1f} points")
print(f"  Accuracy at this threshold: {best_accuracy * 100:.1f}%")
print(f"  Sensitivity (PD caught): {best_sensitivity * 100:.1f}%")
print(f"  Specificity (Healthy correct): {best_specificity * 100:.1f}%")

# ============================================================
# STEP 7: Demo - Score specific patients
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Diagnosing Patient(s)")
print("=" * 70)

print(f"\n  Threshold: {best_threshold:.1f} points")
print(f"  Rule: Score >= {best_threshold:.1f} -> Predict Parkinson's")
print(f"        Score <  {best_threshold:.1f} -> Predict Not Parkinson's\n")

import argparse
parser = argparse.ArgumentParser(description="Questionnaire Weightage & Diagnosis")
parser.add_argument('--id', type=int, default=None, help='Specific Patient ID to diagnose')
parser.add_argument('--mismatches', action='store_true', help='Print all mismatched patients')
args, unknown = parser.parse_known_args()

def print_report(demo_pid):
    if demo_pid in patient_ids:
        idx = patient_ids.index(demo_pid)
        score = patient_scores[idx]
        prediction = "Parkinson's" if score >= best_threshold else "Not Parkinson's"
        actual = patient_conditions[idx]

        # Calculate confidence percentage based on distance from threshold
        if score >= best_threshold:
            confidence = min(100.0, 50.0 + 50.0 * (score - best_threshold) / (100.0 - best_threshold))
        else:
            confidence = min(100.0, 50.0 + 50.0 * (best_threshold - score) / best_threshold)

        # Show which questions contributed
        true_questions = np.where(X[idx] == 1)[0]
        contributing = [(question_texts[q][:50], point_values[q]) for q in true_questions]
        contributing.sort(key=lambda x: x[1], reverse=True)

        match_status = "MATCH (Correct)" if (prediction == "Parkinson's" and actual == "Parkinson's") or (prediction == "Not Parkinson's" and actual != "Parkinson's") else "MISMATCH"

        # Generate Explanation
        if prediction == "Parkinson's":
            if contributing:
                top_q = contributing[0][0].strip()
                explanation = f"Score ({score:.1f} pts) exceeded the risk threshold ({best_threshold:.1f} pts).\n                           This was primarily driven by symptoms such as:\n                           '{top_q}'."
            else:
                explanation = f"Score ({score:.1f} pts) exceeded the risk threshold ({best_threshold:.1f} pts)."
        else:
            explanation = f"Score ({score:.1f} pts) remained below the risk threshold ({best_threshold:.1f} pts),\n                           indicating an absence of major Parkinson's-related symptoms."

        print(f"\n  DIAGNOSIS REPORT FOR PATIENT ID: {demo_pid:03d}")
        print(f"  --------------------------------------------------")
        print(f"  AI Prediction (Blind):   {prediction}  (Confidence: {confidence:.1f}%)")
        print(f"  Total Score:             {score:.1f} pts")
        print(f"  Actual Ground Truth:     {actual}")
        print(f"  Verification Result:     {match_status}")
        print(f"  Explanation:             {explanation}")
        print(f"  --------------------------------------------------")
        print(f"  Contributing Questions (Answered 'True'):")
        
        if contributing:
            for q_text, pts in contributing[:5]:  # Show top 5 contributing questions
                print(f"    + {pts:>4.1f} pts: {q_text}")
            if len(contributing) > 5:
                print(f"    ... and {len(contributing)-5} more.")
        else:
            print(f"    (No 'True' answers)")
    else:
        print(f"Error: Patient ID {demo_pid} not found in PADS dataset.")

if __name__ == "__main__":
    if args.id is not None:
        print_report(args.id)
    elif args.mismatches:
        print("\n  ==================================================")
        print("  EVALUATING ALL PATIENTS FOR MISMATCHES")
        print("  ==================================================")
        mismatches = []
        for idx, pid in enumerate(patient_ids):
            score = patient_scores[idx]
            prediction = "Parkinson's" if score >= best_threshold else "Not Parkinson's"
            actual = patient_conditions[idx]
            is_correct = (prediction == "Parkinson's" and actual == "Parkinson's") or (prediction == "Not Parkinson's" and actual != "Parkinson's")
            if not is_correct:
                mismatches.append((pid, actual, prediction, score))
                
        print(f"  Found {len(mismatches)} mismatches out of {len(patient_ids)} patients.")
        print(f"  {'PID':<5} | {'Actual Condition':<30} | {'AI Predicted':<17} | {'Score'}")
        print("  " + "-" * 75)
        for m in mismatches:
            print(f"  {m[0]:<5} | {m[1]:<30} | {m[2]:<17} | {m[3]:.1f} pts")
        print("\n")
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
    
    # ============================================================
    # STEP 8: Final Summary
    # ============================================================
    print("=" * 70)
    print("FINAL SUMMARY: How the Algorithm Works")
    print("=" * 70)
    print(f"""
      1. Each of the 30 questionnaire questions has a POINT VALUE
         (ranging from 0 to 10 points) based on how strongly it
         correlates with Parkinson's Disease.
    
      2. For a new patient, add up the points for every question
         where they answered TRUE.
    
      3. If the total score is >= {best_threshold:.1f} points, predict PARKINSON'S.
         If the total score is <  {best_threshold:.1f} points, predict NOT PARKINSON'S.
    
      4. This simple scoring system achieves {best_accuracy * 100:.1f}% accuracy
         on the dataset of {len(patient_ids)} patients.
    
      The TOP 5 highest-weighted questions are:
    """)
    
    for rank in range(5):
        idx = combined_sorted[rank]
        q_text = question_texts[idx]
        if len(q_text) > 55:
            q_text = q_text[:52] + '...'
        print(f"    {rank+1}. ({point_values[idx]:.1f} pts) {q_text}")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
