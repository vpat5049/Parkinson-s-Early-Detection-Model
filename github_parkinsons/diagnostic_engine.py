import sys
import os
import json
import glob
import numpy as np

# Ensure Algorithms/Main Algorithms is in python path
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ALGO_DIR = os.path.join(CUR_DIR, 'Algorithms', 'Main Algorithms')
if ALGO_DIR not in sys.path:
    sys.path.insert(0, ALGO_DIR)

DATASET_DIR = os.path.join(CUR_DIR, 'pads-parkinsons-disease-smartwatch-dataset-1.0.0')

CLASS_NAMES = ["Healthy", "Parkinson's", "Other Movement Disorders"]

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
TOTAL_ACCURACY = sum(ACCURACIES.values())
WEIGHTS = {k: v / TOTAL_ACCURACY for k, v in ACCURACIES.items()}

_LOADED_MODULES = None

def load_ensemble_models():
    """
    Loads all 12 sub-models once in background, suppressing terminal noise.
    """
    global _LOADED_MODULES
    if _LOADED_MODULES is not None:
        return _LOADED_MODULES

    # Save original stdout / stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    try:
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

        _LOADED_MODULES = {
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
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return _LOADED_MODULES


def get_patient_list():
    """
    Returns sorted list of all available patients with their ground-truth condition.
    """
    patients_dir = os.path.join(DATASET_DIR, 'patients')
    patient_files = glob.glob(os.path.join(patients_dir, 'patient_*.json'))
    
    patients = []
    for fpath in patient_files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            pid = int(data.get('id', 0))
            condition = data.get('condition', 'Unknown')
            patients.append({
                "id": pid,
                "condition": condition,
                "label": f"Patient {pid:03d} — {condition}"
            })
        except Exception:
            continue

    patients.sort(key=lambda p: p["id"])
    return patients


def get_patient_metadata(pid):
    """
    Loads detailed clinical info from patient_{pid:03d}.json
    """
    pfile = os.path.join(DATASET_DIR, 'patients', f'patient_{pid:03d}.json')
    if os.path.exists(pfile):
        with open(pfile, 'r') as f:
            return json.load(f)
    return {}


def get_patient_questionnaire(pid):
    """
    Loads responses from questionnaire_response_{pid:03d}.json
    """
    qfile = os.path.join(DATASET_DIR, 'questionnaire', f'questionnaire_response_{pid:03d}.json')
    if os.path.exists(qfile):
        with open(qfile, 'r') as f:
            data = json.load(f)
            return data.get('item', [])
    return []


def run_master_diagnosis(pid):
    """
    Runs the 12-model weighted ensemble inference for a specific patient.
    Returns a comprehensive diagnosis dictionary.
    """
    modules = load_ensemble_models()

    total_valid_weight = 0.0
    weighted_probs = np.array([0.0, 0.0, 0.0])
    model_votes = []

    for name, mod in modules.items():
        w = WEIGHTS[name]
        if pid in mod.patient_ids:
            idx = mod.patient_ids.index(pid)
            feat = mod.X_scaled[idx] if hasattr(mod, 'X_scaled') else mod.X[idx]
            prob_arr = mod.clf.predict_proba([feat])[0]
            
            # Handle edge cases where class length < 3
            if len(prob_arr) < 3:
                full_probs = [0.0, 0.0, 0.0]
                for i, c in enumerate(mod.clf.classes_):
                    full_probs[c] = prob_arr[i]
                prob_arr = np.array(full_probs)

            vote_idx = int(np.argmax(prob_arr))
            vote_class = CLASS_NAMES[vote_idx]
            conf = float(prob_arr[vote_idx])

            weighted_probs += (prob_arr * w)
            total_valid_weight += w

            model_votes.append({
                "task": name,
                "vote": vote_class,
                "confidence": conf,
                "raw_weight": w,
                "probabilities": {CLASS_NAMES[i]: float(prob_arr[i]) for i in range(3)},
                "has_data": True
            })
        else:
            model_votes.append({
                "task": name,
                "vote": "Missing Data",
                "confidence": 0.0,
                "raw_weight": w,
                "probabilities": {c: 0.0 for c in CLASS_NAMES},
                "has_data": False
            })

    if total_valid_weight > 0:
        final_probs = weighted_probs / total_valid_weight
        winning_idx = int(np.argmax(final_probs))
        prediction = CLASS_NAMES[winning_idx]
        confidence = float(final_probs[winning_idx])
    else:
        final_probs = np.array([0.33, 0.33, 0.33])
        prediction = "Unknown (No Data)"
        confidence = 0.0

    # Add normalized effective voting weights to results
    for vote in model_votes:
        if vote["has_data"] and total_valid_weight > 0:
            vote["effective_weight"] = (vote["raw_weight"] / total_valid_weight) * 100.0
        else:
            vote["effective_weight"] = 0.0

    # Sort votes by effective weight (highest impact first)
    model_votes.sort(key=lambda v: v["effective_weight"], reverse=True)

    # Fetch ground truth
    meta = get_patient_metadata(pid)
    actual = meta.get("condition", "Unknown")

    # Match evaluation
    other_disorders = ["Essential Tremor", "Multiple Sclerosis", "Atypical Parkinsonism", "Other Movement Disorders"]
    if prediction == actual:
        is_match = True
    elif prediction == "Other Movement Disorders" and actual in other_disorders:
        is_match = True
    else:
        is_match = False

    return {
        "pid": pid,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "Healthy": float(final_probs[0]),
            "Parkinson's": float(final_probs[1]),
            "Other Movement Disorders": float(final_probs[2])
        },
        "actual": actual,
        "is_match": is_match,
        "match_status": "MATCH (Correct)" if is_match else "MISMATCH",
        "model_votes": model_votes,
        "metadata": meta,
        "questionnaire": get_patient_questionnaire(pid)
    }
