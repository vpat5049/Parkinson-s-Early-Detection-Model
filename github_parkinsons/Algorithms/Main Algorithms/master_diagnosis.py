import sys
import os
import argparse
import numpy as np

# Setup argument parser
parser = argparse.ArgumentParser(description="Master Algorithm: Combines 12 Diagnostic Tests (3-Class)")
parser.add_argument("--id", type=int, help="Patient ID to diagnose (e.g. 21)")
args = parser.parse_args()

print("=" * 80)
print("  INITIALIZING MASTER ALGORITHM (3-CLASS AI)")
print("  (Training 12 specialized Random Forests in the background...)")
print("  Please wait ~30 seconds...")
print("=" * 80)

# Suppress stdout to prevent the 12 scripts from printing their individual training logs
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

# Import all 12 modules
import questionnaire_weightage as m_quest
import resting_wrist_weightage as m_rest
import drinkglas_weightage as m_drink
import touchindex_weightage as m_touchindex
import touchnose_weightage as m_touchnose
import entrainment_weightage as m_entrain
import pointfinger_weightage as m_point
import relaxedtask_weightage as m_relax
import lifthold_weightage as m_lift
import stretchhold_weightage as m_stretch
import holdweight_weightage as m_hold
import crossarms_weightage as m_cross

sys.stdout.close()
sys.stdout = old_stdout

print("  [SUCCESS] All 12 models successfully loaded and trained!\n")

ACCURACIES = {
    "Questionnaire": 77.0,
    "Resting Wrist": 75.0,
    "Drink Glass": 70.0,
    "Touch Index": 65.7,
    "Entrainment": 65.0,
    "Relaxed Task": 65.0,
    "Touch Nose": 63.0,
    "Point Finger": 60.0,
    "Lift Hold": 60.0,
    "Stretch Hold": 60.0,
    "Hold Weight": 60.0,
    "Cross Arms": 60.0
}

total_accuracy = sum(ACCURACIES.values())
WEIGHTS = {k: v / total_accuracy for k, v in ACCURACIES.items()}

MODULES = {
    "Questionnaire": m_quest,
    "Resting Wrist": m_rest,
    "Drink Glass": m_drink,
    "Touch Index": m_touchindex,
    "Entrainment": m_entrain,
    "Relaxed Task": m_relax,
    "Touch Nose": m_touchnose,
    "Point Finger": m_point,
    "Lift Hold": m_lift,
    "Stretch Hold": m_stretch,
    "Hold Weight": m_hold,
    "Cross Arms": m_cross
}

def get_probabilities(module_name, pid):
    """
    Extracts the 3-class probability array [Healthy, Parkinson's, Other]
    """
    mod = MODULES[module_name]
    if pid not in mod.patient_ids:
        return None
        
    idx = mod.patient_ids.index(pid)
    feat = mod.X_scaled[idx] if hasattr(mod, 'X_scaled') else mod.X[idx]
    
    # Returns [prob_healthy, prob_pd, prob_other]
    prob_array = mod.clf.predict_proba([feat])[0]
    
    # In case a model only saw 2 classes during a split (rare, but possible if missing data), 
    # we enforce a shape of 3. But all datasets have all 3 classes globally.
    if len(prob_array) < 3:
        # Fallback padding if needed
        classes = mod.clf.classes_
        full_probs = [0.0, 0.0, 0.0]
        for i, c in enumerate(classes):
            full_probs[c] = prob_array[i]
        return np.array(full_probs)
        
    return prob_array

def master_diagnose(pid):
    print("=" * 80)
    print(f"  MASTER DIAGNOSIS REPORT FOR PATIENT ID: {pid:03d}")
    print("=" * 80)
    
    total_valid_weight = 0.0
    weighted_probs = np.array([0.0, 0.0, 0.0]) # [Healthy, PD, Other]
    
    results = []
    
    for name in MODULES.keys():
        prob_array = get_probabilities(name, pid)
        if prob_array is not None:
            w = WEIGHTS[name]
            weighted_probs += (prob_array * w)
            total_valid_weight += w
            results.append((name, prob_array, w))
        else:
            results.append((name, None, 0.0))
            
    if total_valid_weight == 0:
        print(f"  ERROR: Patient {pid} has no data across any of the 12 tests.")
        return
        
    # Calculate final normalized probabilities
    final_probs = weighted_probs / total_valid_weight
    
    # Determine ultimate prediction
    class_names = ["Healthy", "Parkinson's", "Other Movement Disorders"]
    winning_class_idx = np.argmax(final_probs)
    prediction = class_names[winning_class_idx]
    confidence = final_probs[winning_class_idx]
    
    # Get ground truth
    actual = "Unknown"
    for mod in MODULES.values():
        if pid in mod.patient_ids:
            idx = mod.patient_ids.index(pid)
            actual = mod.patient_conditions[idx]
            break
            
    match_status = "MATCH (Correct)" if (prediction == actual) or (prediction == "Other Movement Disorders" and actual in ["Essential Tremor", "Multiple Sclerosis", "Atypical Parkinsonism", "Other Movement Disorders"]) else "MISMATCH"
    
    print(f"  FINAL MASTER PREDICTION:  {prediction}   (Confidence: {confidence*100:.1f}%)")
    print(f"  Actual Ground Truth:      {actual}")
    print(f"  Verification Result:      {match_status}")
    print("-" * 80)
    print("  HOW THE 12 MODELS VOTED (Weighted Ensemble):")
    print("-" * 80)
    print(f"  {'Task Name':<20} | {'Prediction':<25} | {'Confidence':<12} | {'Voting Weight':<15}")
    print("  " + "-" * 75)
    
    # Sort results by weight (most important first)
    results.sort(key=lambda x: x[2], reverse=True)
    
    for name, prob_array, w in results:
        if prob_array is not None:
            vote_idx = np.argmax(prob_array)
            vote = class_names[vote_idx]
            conf_str = f"{prob_array[vote_idx]*100:.1f}%"
            weight_str = f"{(w/total_valid_weight)*100:.1f}%"
            print(f"  {name:<20} | {vote:<25} | {conf_str:<12} | {weight_str:<15}")
        else:
            print(f"  {name:<20} | {'MISSING DATA':<25} | {'---':<12} | {'0.0%':<15}")

    print("=" * 80)
    print("\n")

if __name__ == "__main__":
    if args.id is not None:
        master_diagnose(args.id)
    else:
        while True:
            try:
                val = input("Enter a Patient ID to run the Master Algorithm (or 'q' to quit): ")
                if val.lower() == 'q':
                    break
                pid = int(val)
                master_diagnose(pid)
            except ValueError:
                print("Please enter a valid number or 'q'.")
            except EOFError:
                break
