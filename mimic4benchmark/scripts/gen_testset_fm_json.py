import json
import os

# Load the patient split JSON
with open(os.path.join(os.path.dirname(__file__), '../resources/mimic-iv-patient-split.json'), 'r') as file:
    patient_splits = json.load(file)

# Combine the splits, marking test patients with a 1 and train with a 0
all_patients = [(patient, 0) for patient in patient_splits['train']] + \
               [(patient, 1) for patient in patient_splits['test']]

# Write to testset.csv
with open(os.path.join(os.path.dirname(__file__), '../resources/testset.csv'), 'w') as out_file:
    for patient_id, is_test in all_patients:
        out_file.write(f"{patient_id},{is_test}\n")
