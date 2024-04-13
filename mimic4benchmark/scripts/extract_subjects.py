from __future__ import absolute_import
from __future__ import print_function

import argparse
import json
import random
import yaml

from mimic4benchmark.mimic4csv import *
from mimic4benchmark.preprocessing import add_hcup_ccs_2015_groups, make_phenotype_label_matrix
from mimic4benchmark.util import dataframe_from_csv

parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-IV CSV files.')
parser.add_argument('mimic4_path', type=str, help='Directory containing MIMIC-IV CSV files.')
parser.add_argument('output_path', type=str, help='Directory where per-subject data should be written.')
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
                    default=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'])
parser.add_argument('--phenotype_definitions', '-p', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/hcup_ccs_2015_definitions.yaml'),
                    help='YAML file with phenotype definitions.')
parser.add_argument('--itemids_file', '-i', type=str, help='CSV containing list of ITEMIDs to keep.')
parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Verbosity in output')
parser.add_argument('--quiet', '-q', dest='verbose', action='store_false', help='Suspend printing of details')
parser.set_defaults(verbose=True)
parser.add_argument('--test', action='store_true', help='TEST MODE: process only 1000 subjects, 1000000 events.')
args, _ = parser.parse_known_args()

try:
    os.makedirs(args.output_path)
except:
    pass

patients = read_patients_table(args.mimic4_path)
admits = read_admissions_table(args.mimic4_path)
stays = read_icustays_table(args.mimic4_path)
if args.verbose:
    print('START:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
          stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

#stays = remove_icustays_with_transfers(stays)
#if args.verbose:
#    print('REMOVE ICU TRANSFERS:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
#          stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

stays = merge_on_subject_admission(stays, admits)
stays = merge_on_subject(stays, patients)
#stays = filter_admissions_on_nb_icustays(stays)
#if args.verbose:
#    print('REMOVE MULTIPLE STAYS PER ADMIT:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
#          stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

#print(patients.columns)
stays = add_age_to_icustays(stays,patients)
stays = add_observation_window_length_to_icustays(stays)
stays = add_inunit_mortality_to_icustays(stays)
stays = add_inhospital_mortality_to_icustays(stays)
stays = filter_icustays_on_age(stays)
if args.verbose:
    print('REMOVE PATIENTS AGE < 18:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
          stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

# exclude patients with no chart or lab events recorded during the stay
def filter_stays_with_events_during_stay(mimic4_path, stays, event_table_names):
    stays_with_events_during_stay = set()

    for table_name in event_table_names:
        event_table_path = os.path.join(mimic4_path, f'{table_name}.csv')

        try:
            event_df = dataframe_from_csv(event_table_path,
                                          usecols=['subject_id', 'hadm_id', 'charttime', 'stay_id'])
            merge_keys = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']
        except ValueError:
            # Fallback if 'icustay_id' is not in the table, merge on 'subject_id' and 'hadm_id' only
            event_df = dataframe_from_csv(event_table_path, usecols=['subject_id', 'hadm_id', 'charttime'])
            merge_keys = ['SUBJECT_ID', 'HADM_ID']

        event_df['CHARTTIME'] = pd.to_datetime(event_df['CHARTTIME'])

        merged_df = pd.merge(stays, event_df, on=merge_keys, how='inner')

        valid_stays = merged_df[(merged_df['CHARTTIME'] >= merged_df['INTIME']) &
                                (merged_df['CHARTTIME'] <= merged_df['OUTTIME'])]
        stays_with_events_during_stay.update(valid_stays['ICUSTAY_ID'].unique())

    # Filter stays to only those with events during the stay
    filtered_stays = stays[stays['ICUSTAY_ID'].isin(stays_with_events_during_stay)]

    return filtered_stays

event_table_names = ['CHARTEVENTS', 'LABEVENTS']
filtered_stays = filter_stays_with_events_during_stay(args.mimic4_path, stays, event_table_names)

if args.verbose:
    print(f'FILTER FOR PATIENTS WITH EVENT RECORDS:\n\tBefore filtering: {stays.shape[0]} stays\n\tAfter filtering: {filtered_stays.shape[0]} stays\n\tNumber of stays removed: {stays.shape[0] - filtered_stays.shape[0]}')

stays = filtered_stays

# filter with mimic-iv-patient-split.json
with open('mimic-iv-patient-split.json', 'r') as file:
    patient_splits = json.load(file)

all_duett_patients = [patient for patient in patient_splits['train']] + \
               [patient for patient in patient_splits['test']]

# TODO: remove this for production
# all_duett_patients = random.sample(all_duett_patients, 1000)

filtered_stays = stays[stays['SUBJECT_ID'].isin(all_duett_patients)]
if args.verbose:
    print(f'FILTER FOR PATIENTS WITH mimic-iv-patient-split.json:\n\tBefore filtering: {stays.shape[0]} stays\n\tAfter filtering: {filtered_stays.shape[0]} stays\n\tNumber of stays removed: {stays.shape[0] - filtered_stays.shape[0]}')
stays = filtered_stays

stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)
diagnoses = read_icd_diagnoses_table(args.mimic4_path)
diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)
count_icd_codes(diagnoses, output_path=os.path.join(args.output_path, 'diagnosis_counts.csv'))

phenotypes = add_hcup_ccs_2015_groups(diagnoses, yaml.load(open(args.phenotype_definitions, 'r'), Loader=yaml.SafeLoader))
make_phenotype_label_matrix(phenotypes, stays).to_csv(os.path.join(args.output_path, 'phenotype_labels.csv'),
                                                      index=False, quoting=csv.QUOTE_NONNUMERIC)

if args.test:
    pat_idx = np.random.choice(patients.shape[0], size=1000)
    patients = patients.iloc[pat_idx]
    stays = stays.merge(patients[['SUBJECT_ID']], left_on='SUBJECT_ID', right_on='SUBJECT_ID')
    args.event_tables = [args.event_tables[0]]
    print('Using only', stays.shape[0], 'stays and only', args.event_tables[0], 'table')

subjects = stays.SUBJECT_ID.unique()
break_up_stays_by_subject(stays, args.output_path, subjects=subjects)
break_up_diagnoses_by_subject(phenotypes, args.output_path, subjects=subjects)
items_to_keep = set(
    [int(itemid) for itemid in dataframe_from_csv(args.itemids_file)['ITEMID'].unique()]) if args.itemids_file else None
for table in args.event_tables:
    read_events_table_and_break_up_by_subject(args.mimic4_path, table, args.output_path, items_to_keep=items_to_keep,
                                              subjects_to_keep=subjects)
