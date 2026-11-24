import sys
import os
import argparse
import numpy as np

# Setup argument parser
parser = argparse.ArgumentParser(description="Master Algorithm: Combines 12 Diagnostic Tests")
parser.add_argument("--id", type=int, help="Patient ID to diagnose (e.g. 21)")
args = parser.parse_args()

print("=" * 80)
print("  INITIALIZING MASTER ALGORITHM")
print("  (Training 12 specialized Random Forests in the background...)")
print("  Please wait ~30 seconds...")
print("=" * 80)

# Suppress stdout to prevent the 12 scripts from printing their individual training logs
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

# Import all 12 modules
import questionnaire_weightage_backup as m_quest
import resting_wrist_weightage_backup as m_rest
import drinkglas_weightage_backup as m_drink
import touchindex_weightage_backup as m_touchindex
import touchnose_weightage_backup as m_touchnose
import entrainment_weightage_backup as m_entrain
import pointfinger_weightage_backup as m_point
import relaxedtask_weightage_backup as m_relax
import lifthold_weightage_backup as m_lift
import stretchhold_weightage_backup as m_stretch
import holdweight_weightage_backup as m_hold
import crossarms_weightage_backup as m_cross

sys.stdout.close()
sys.stdout = old_stdout

print("  [SUCCESS] All 12 models successfully loaded and trained!\n")

# Define base mathematical accuracies for each model (used to weight their votes)
# These were calculated via cross-validation during the optimization phase.
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

# Convert absolute accuracies into percentage weights that sum to 1.0 (100%)
total_accuracy = sum(ACCURACIES.values())
WEIGHTS = {k: v / total_accuracy for k, v in ACCURACIES.items()}

# Map names to their imported module objects
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

def get_probability(module_name, pid):
    """
    Extracts the Parkinson's probability (0.0 to 1.0) from a specific module for a given patient.
    """
    mod = MODULES[module_name]
    
    # Check if patient exists in this module's data
    if pid not in mod.patient_ids:
        return None
        
    idx = mod.patient_ids.index(pid)
    
    # Questionnaire uses a point-based scoring system
    if module_name == "Questionnaire":
        score = mod.patient_scores[idx]
        thresh = mod.best_threshold
        # Convert score distance from threshold into a probability between 0 and 1
        if score >= thresh:
            prob = min(1.0, 0.5 + 0.5 * ((score - thresh) / (100.0 - thresh)))
        else:
            prob = max(0.0, 0.5 - 0.5 * ((thresh - score) / thresh))
        return prob
        
    # All other motion tasks use the Random Forest predict_proba
    else:
        # Some modules scale their data, some don't. Try to use X_scaled if available.
        feat = mod.X_scaled[idx] if hasattr(mod, 'X_scaled') else mod.X[idx]
        prob_pd = mod.clf.predict_proba([feat])[0][1] # [1] is the probability of Parkinson's (Class 1)
        return prob_pd

def master_diagnose(pid):
    print("=" * 80)
    print(f"  MASTER DIAGNOSIS REPORT FOR PATIENT ID: {pid:03d}")
    print("=" * 80)
    
    # To calculate the weighted average, we must track total weight (in case a patient is missing a test)
    total_valid_weight = 0.0
    weighted_probability_sum = 0.0
    
    results = []
    
    for name in MODULES.keys():
        prob = get_probability(name, pid)
        if prob is not None:
            w = WEIGHTS[name]
            weighted_probability_sum += (prob * w)
            total_valid_weight += w
            results.append((name, prob, w))
        else:
            results.append((name, None, 0.0))
            
    if total_valid_weight == 0:
        print(f"  ERROR: Patient {pid} has no data across any of the 12 tests.")
        return
        
    # Calculate final normalized probability
    final_prob = weighted_probability_sum / total_valid_weight
    
    # Get ground truth from any available module
    actual = "Unknown"
    for mod in MODULES.values():
        if pid in mod.patient_ids:
            idx = mod.patient_ids.index(pid)
            actual = mod.patient_conditions[idx]
            break
            
    prediction = "Parkinson's" if final_prob >= 0.5 else "Not Parkinson's"
    match_status = "MATCH (Correct)" if (prediction == "Parkinson's" and actual == "Parkinson's") or (prediction == "Not Parkinson's" and actual != "Parkinson's") else "MISMATCH"
    
    print(f"  FINAL MASTER PREDICTION:  {prediction}   (Confidence: {final_prob*100:.1f}%)")
    print(f"  Actual Ground Truth:      {actual}")
    print(f"  Verification Result:      {match_status}")
    print("-" * 80)
    print("  HOW THE 12 MODELS VOTED (Weighted Ensemble):")
    print("-" * 80)
    print(f"  {'Task Name':<20} | {'Prediction':<17} | {'Confidence':<12} | {'Voting Weight':<15}")
    print("  " + "-" * 75)
    
    # Sort results by weight (most important first)
    results.sort(key=lambda x: x[2], reverse=True)
    
    for name, prob, w in results:
        if prob is not None:
            vote = "Parkinson's" if prob >= 0.5 else "Not Parkinson's"
            conf_str = f"{prob*100:.1f}%"
            weight_str = f"{(w/total_valid_weight)*100:.1f}%"
            print(f"  {name:<20} | {vote:<17} | {conf_str:<12} | {weight_str:<15}")
        else:
            print(f"  {name:<20} | {'MISSING DATA':<17} | {'---':<12} | {'0.0%':<15}")

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
