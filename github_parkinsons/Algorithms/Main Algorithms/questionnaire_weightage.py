import os
import json
import glob
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Setup argument parser
parser = argparse.ArgumentParser(description="Evaluate a patient's questionnaire.")
parser.add_argument("--id", type=int, help="Patient ID to diagnose (e.g. 21)")
parser.add_argument("--mismatches", action="store_true", help="Evaluate all patients and print mismatches")
args = parser.parse_args()

BASE_DIR = 'pads-parkinsons-disease-smartwatch-dataset-1.0.0'
QUESTIONNAIRE_DIR = os.path.join(BASE_DIR, 'questionnaire')
PATIENTS_DIR = os.path.join(BASE_DIR, 'patients')

# Load files
quest_files = glob.glob(os.path.join(QUESTIONNAIRE_DIR, 'questionnaire_response_*.json'))

X = []
y = []
patient_ids = []
patient_conditions = []

# Question texts (mapping from standard PADS keys if needed, but we can just use indexes)
# For simplicity, we just extract the boolean values for the 30 questions in sorted key order
question_texts = []

for file in quest_files:
    pid_str = os.path.basename(file).split('_')[-1].split('.')[0]
    pid = int(pid_str)
    
    with open(file, 'r') as f:
        q_data = json.load(f)
        
    # Get condition
    patient_file = os.path.join(PATIENTS_DIR, f'patient_{pid:03d}.json')
    if not os.path.exists(patient_file):
        continue
        
    with open(patient_file, 'r') as f:
        p_data = json.load(f)
        condition = p_data.get('condition', 'Unknown')
    
    # Extract the 30 questions from the 'item' array
    items = q_data.get('item', [])
    features = []
    
    if not question_texts:
        question_texts = [item.get('text', f"Q{item.get('link_id', '')}") for item in items]
        
    for item in items:
        val = item.get('answer', False)
        if isinstance(val, bool):
            features.append(1 if val else 0)
        elif isinstance(val, str):
            features.append(1 if val.lower() == 'true' or val.lower() == 'yes' else 0)
        else:
            features.append(0)
            
    if len(features) > 0:
        X.append(features)
        
        # 3-Class logic
        if condition == "Parkinson's":
            y.append(1)
        elif condition == "Healthy" or condition == "Healthy Control":
            y.append(0)
        else:
            y.append(2)
            
        patient_ids.append(pid)
        patient_conditions.append(condition)

X = np.array(X)
y = np.array(y)

# Train the Random Forest
clf = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced')
clf.fit(X, y)

def print_report(demo_pid):
    if demo_pid in patient_ids:
        idx = patient_ids.index(demo_pid)
        patient_feat = X[idx]
        actual = patient_conditions[idx]
        
        pred_class = clf.predict([patient_feat])[0]
        probs = clf.predict_proba([patient_feat])[0]
        
        if pred_class == 0:
            prediction = "Healthy"
        elif pred_class == 1:
            prediction = "Parkinson's"
        else:
            prediction = "Other Movement Disorders"
            
        prob_pd = max(probs)
        
        match_status = "MATCH (Correct)" if (prediction == actual) or (prediction == "Other Movement Disorders" and actual in ["Essential Tremor", "Multiple Sclerosis", "Atypical Parkinsonism", "Other Movement Disorders"]) else "MISMATCH"

        print(f"\n  DIAGNOSIS REPORT FOR PATIENT ID: {demo_pid:03d}")
        print(f"  --------------------------------------------------")
        print(f"  AI Prediction (Blind):   {prediction}  (Confidence: {prob_pd*100:.1f}%)")
        print(f"  Actual Ground Truth:     {actual}")
        print(f"  Verification Result:     {match_status}")
        print(f"  --------------------------------------------------")
    else:
        print(f"Error: Patient ID {demo_pid} not found in dataset.")

if __name__ == "__main__":
    if args.id is not None:
        print_report(args.id)
    elif args.mismatches:
        print("\n  ==================================================")
        print("  EVALUATING ALL PATIENTS FOR MISMATCHES")
        print("  ==================================================")
        mismatches = []
        for idx, pid in enumerate(patient_ids):
            pred_class = clf.predict([X[idx]])[0]
            if pred_class == 0:
                prediction = "Healthy"
            elif pred_class == 1:
                prediction = "Parkinson's"
            else:
                prediction = "Other Movement Disorders"
                
            actual = patient_conditions[idx]
            is_correct = (prediction == actual) or (prediction == "Other Movement Disorders" and actual in ["Essential Tremor", "Multiple Sclerosis", "Atypical Parkinsonism", "Other Movement Disorders"])
            if not is_correct:
                mismatches.append((pid, actual, prediction))
                
        print(f"  Found {len(mismatches)} mismatches out of {len(patient_ids)} patients.")
    else:
        while True:
            try:
                val = input("Which patient ID would you like to check? (e.g. 94, or 'q' to quit): ")
                if val.lower() == 'q':
                    break
                pid = int(val)
                print_report(pid)
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")
            except EOFError:
                break
