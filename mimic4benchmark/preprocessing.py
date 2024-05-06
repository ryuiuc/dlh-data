from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re

from pandas import DataFrame, Series, concat, NA, isna, to_numeric, set_option

from mimic4benchmark.util import dataframe_from_csv

###############################
# Non-time series preprocessing
###############################

g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}


def transform_gender(gender_series):
    global g_map
    return {'Gender': gender_series.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map['OTHER'])}

def transform_english_language(english_language_series):
    el_map = {True: 1, False: 0, 'OTHER': 3, '': 4}
    return {'English Language': english_language_series.fillna('').apply(lambda s: el_map[s] if s in el_map else el_map['OTHER'])}

def transform_marital_status(marital_status_series):
    ms_map = {'SINGLE': 1, 'WIDOWED': 2, 'MARRIED': 3, 'DIVORCED': 4, 'OTHER': 5, '': 6}
    return {'Marital Status':marital_status_series.fillna('').apply(lambda s: ms_map[s] if s in ms_map else ms_map['OTHER'])}

def transform_insurance(insurance_series):
    insurance_map = {'Medicare': 1, 'Other': 2, 'Medicaid': 3, 'OTHER': 4, '': 5}
    return {'Insurance':insurance_series.fillna('').apply(lambda s: insurance_map[s] if s in insurance_map else insurance_map['OTHER'])}

def transform_admission_location(admission_location_series):
    admission_location_map = {
        'PACU': 1, 'PROCEDURE SITE': 2, 'EMERGENCY ROOM': 3, 'PHYSICIAN REFERRAL': 4,
        'TRANSFER FROM SKILLED NURSING FACILITY': 5, 'CLINIC REFERRAL': 6,
        'TRANSFER FROM HOSPITAL': 7, 'WALK-IN/SELF REFERRAL': 8,
        'AMBULATORY SURGERY TRANSFER': 9, 'INFORMATION NOT AVAILABLE': 10,
        'INTERNAL TRANSFER TO OR FROM PSYCH': 11,
        'OTHER': 12, '': 13
    }
    return {'Admission Location':admission_location_series.fillna('').apply(lambda s: admission_location_map[s] if s in admission_location_map else admission_location_map['OTHER'])}

def transform_admission_type(admission_type_series):
    admission_type_map = {
        'OBSERVATION ADMIT': 1, 'EW EMER.': 2, 'DIRECT EMER.': 3, 'ELECTIVE': 4,
        'DIRECT OBSERVATION': 5, 'URGENT': 6, 'EU OBSERVATION': 7,
        'SURGICAL SAME DAY ADMISSION': 8, 'AMBULATORY OBSERVATION': 9, 'OTHER': 10, '': 11
    }
    return {'Admission Type':admission_type_series.fillna('').apply(lambda s: admission_type_map[s] if s in admission_type_map else admission_type_map['OTHER'])}


def transform_first_care_unit(first_care_unit_series):
    first_care_unit_map = {
        'Trauma SICU (TSICU)': 1,
        'Medical/Surgical Intensive Care Unit (MICU/SICU)': 2,
        'Coronary Care Unit (CCU)': 3,
        'Neuro Surgical Intensive Care Unit (Neuro SICU)': 4,
        'Neuro Stepdown': 5,
        'Medical Intensive Care Unit (MICU)': 6,
        'Neuro Intermediate': 7,
        'Surgical Intensive Care Unit (SICU)': 8,
        'Cardiac Vascular Intensive Care Unit (CVICU)': 9,
        'OTHER': 10, '': 11
    }
    return {'First Care Unit':first_care_unit_series.fillna('').apply(lambda s: first_care_unit_map[s] if s in first_care_unit_map else first_care_unit_map['OTHER'])}


e_map = {'ASIAN': 1,
         'BLACK': 2,
         'CARIBBEAN ISLAND': 2,
         'HISPANIC': 3,
         'SOUTH AMERICAN': 3,
         'WHITE': 4,
         'MIDDLE EASTERN': 4,
         'PORTUGUESE': 4,
         'AMERICAN INDIAN': 0,
         'NATIVE HAWAIIAN': 0,
         'UNABLE TO OBTAIN': 0,
         'PATIENT DECLINED TO ANSWER': 0,
         'UNKNOWN': 0,
         'OTHER': 0,
         '': 0}


def transform_ethnicity(ethnicity_series):
    global e_map

    def aggregate_ethnicity(ethnicity_str):
        return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]

    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return {'Ethnicity': ethnicity_series.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map['OTHER'])}


def assemble_episodic_data(stays, diagnoses):
    data = {'Icustay': stays.ICUSTAY_ID, 'Age': stays.AGE, 'Length of Stay': stays.LOS,
            'Mortality': stays.MORTALITY, 'English Language': stays.LANGUAGE, 'Marital Status': stays.MARITAL_STATUS,
            'Insurance': stays.INSURANCE, 'Admission Location': stays.ADMISSION_LOCATION,
            'Admission Type': stays.ADMISSION_TYPE, 'First Care Unit': stays.FIRST_CAREUNIT,
            'Observation Window Length': stays.OBSERVATION_WINDOW_LENGTH}
    data.update(transform_gender(stays.GENDER))
    data.update(transform_ethnicity(stays.RACE))
    data.update(transform_english_language(stays.LANGUAGE))
    data.update(transform_marital_status(stays.MARITAL_STATUS))
    data.update(transform_insurance(stays.INSURANCE))
    data.update(transform_admission_location(stays.ADMISSION_LOCATION))
    data.update(transform_admission_type(stays.ADMISSION_TYPE))
    data.update(transform_first_care_unit(stays.FIRST_CAREUNIT))
    data['Height'] = np.nan
    data['Weight'] = np.nan
    data = DataFrame(data).set_index('Icustay')
    data = data[['Ethnicity', 'Gender', 'Age', 'Height', 'Weight', 'Length of Stay',
                 'English Language', 'Marital Status', 'Insurance', 'Admission Location', 'Admission Type',
                 'First Care Unit', 'Observation Window Length', 'Mortality']]
    return data.merge(extract_diagnosis_labels(diagnoses), left_index=True, right_index=True)


diagnosis_labels = ['4019', '4280', '41401', '42731', '25000', '5849', '2724', '51881', '53081', '5990', '2720',
                    '2859', '2449', '486', '2762', '2851', '496', 'V5861', '99592', '311', '0389', '5859', '5070',
                    '40390', '3051', '412', 'V4581', '2761', '41071', '2875', '4240', 'V1582', 'V4582', 'V5867',
                    '4241', '40391', '78552', '5119', '42789', '32723', '49390', '9971', '2767', '2760', '2749',
                    '4168', '5180', '45829', '4589', '73300', '5845', '78039', '5856', '4271', '4254', '4111',
                    'V1251', '30000', '3572', '60000', '27800', '41400', '2768', '4439', '27651', 'V4501', '27652',
                    '99811', '431', '28521', '2930', '7907', 'E8798', '5789', '79902', 'V4986', 'V103', '42832',
                    'E8788', '00845', '5715', '99591', '07054', '42833', '4275', '49121', 'V1046', '2948', '70703',
                    '2809', '5712', '27801', '42732', '99812', '4139', '3004', '2639', '42822', '25060', 'V1254',
                    '42823', '28529', 'E8782', '30500', '78791', '78551', 'E8889', '78820', '34590', '2800', '99859',
                    'V667', 'E8497', '79092', '5723', '3485', '5601', '25040', '570', '71590', '2869', '2763', '5770',
                    'V5865', '99662', '28860', '36201', '56210']

def extract_diagnosis_labels(diagnoses):
    global diagnosis_labels
    diagnoses['VALUE'] = 1
    labels = diagnoses[['ICUSTAY_ID', 'ICD9_CODE', 'VALUE']].drop_duplicates()\
                      .pivot(index='ICUSTAY_ID', columns='ICD9_CODE', values='VALUE').fillna(0).astype(int)
    #for l in diagnosis_labels:
    #    if l not in labels:
    #        labels[l] = 0
    missing_labels = [label for label in diagnosis_labels if label not in labels.columns]
    missing_columns_df = DataFrame(0, index=labels.index, columns=missing_labels)
    labels = concat([labels, missing_columns_df], axis=1)

    labels = labels[diagnosis_labels]
    return labels.rename(dict(zip(diagnosis_labels, ['Diagnosis ' + d for d in diagnosis_labels])), axis=1)


def add_hcup_ccs_2015_groups(diagnoses, definitions):
    def_map = {}
    for dx in definitions:
        for code in definitions[dx]['codes']:
            def_map[code] = (dx, definitions[dx]['use_in_benchmark'])
    diagnoses['HCUP_CCS_2015'] = diagnoses.ICD9_CODE.apply(lambda c: def_map[c][0] if c in def_map else None)
    diagnoses['USE_IN_BENCHMARK'] = diagnoses.ICD9_CODE.apply(lambda c: int(def_map[c][1]) if c in def_map else None)
    return diagnoses


def make_phenotype_label_matrix(phenotypes, stays=None):
    phenotypes = phenotypes[['ICUSTAY_ID', 'HCUP_CCS_2015']].loc[phenotypes.USE_IN_BENCHMARK > 0].drop_duplicates()
    phenotypes['VALUE'] = 1
    phenotypes = phenotypes.pivot(index='ICUSTAY_ID', columns='HCUP_CCS_2015', values='VALUE')
    if stays is not None:
        phenotypes = phenotypes.reindex(stays.ICUSTAY_ID.sort_values())
    return phenotypes.fillna(0).astype(int).sort_index(axis=0).sort_index(axis=1)


###################################
# Time series preprocessing
###################################

additional_vars = {
220045: "Heart Rate",
220277: "O2 saturation pulseoxymetry",
220210: "Respiratory Rate",
220739: "GCS - Eye Opening",
223900: "GCS - Verbal Response",
223901: "GCS - Motor Response",
224641: "Alarms On",
224168: "Parameters Checked",
220047: "Heart Rate Alarm - Low",
220046: "Heart rate Alarm - High",
220181: "Non Invasive Blood Pressure mean",
220179: "Non Invasive Blood Pressure systolic",
220180: "Non Invasive Blood Pressure diastolic",
223770: "O2 Saturation Pulseoxymetry Alarm - Low",
223769: "O2 Saturation Pulseoxymetry Alarm - High",
224161: "Resp Alarm - High",
224162: "Resp Alarm - Low",
224054: "Braden Sensory Perception",
224057: "Braden Mobility",
224055: "Braden Moisture",
224056: "Braden Activity",
224058: "Braden Nutrition",
224059: "Braden Friction/Shear",
226253: "SpO2 Desat Limit",
223761: "Temperature Fahrenheit",
227344: "IV/Saline lock",
227345: "Gait/Transferring",
227343: "Ambulatory aid",
227346: "Mental status",
227342: "Secondary diagnosis",
227341: "History of falling (within 3 mnths)",
227442: "Potassium (serum)",
220645: "Sodium (serum)",
220602: "Chloride (serum)",
220615: "Creatinine (serum)",
225624: "BUN",
227443: "HCO3 (serum)",
227073: "Anion gap",
220545: "Hematocrit (serum)",
220621: "Glucose (serum)",
220228: "Hemoglobin",
227457: "Platelet Count",
220546: "WBC",
220635: "Magnesium",
223752: "Non-Invasive Blood Pressure Alarm - Low",
223751: "Non-Invasive Blood Pressure Alarm - High",
225677: "Phosphorous",
225625: "Calcium non-ionized",
223791: "Pain Level",
228096: "Richmond-RAS Scale",
227465: "Prothrombin time",
227467: "INR",
227466: "PTT-CHART",
223951: "Capillary Refill R",
224308: "Capillary Refill L",
226531: "Admission Weight (lbs.)",
228299: "Goal Richmond-RAS Scale",
228305: "ST Segment Monitoring On",
223834: "O2 Flow",
225664: "Glucose finger stick (range 70-100)",
224409: "Pain Level Response",
225103: "Intravenous / IV access prior to admission",
227368: "20 Gauge Dressing Occlusive",
228412: "Strength R Arm",
228409: "Strength L Arm",
228411: "Strength R Leg",
228410: "Strength L Leg",
226138: "20 Gauge placed in outside facility",
228236: "Insulin pump",
225092: "Self ADL",
228100: "20 Gauge placed in the field",
225094: "History of slips / falls",
227349: "High risk (.51) interventions",
225668: "Lactic Acid",
228648: "Home TF",
225106: "ETOH",
228649: "Pressure Ulcer Present",
225118: "Difficulty swallowing",
227367: "18 Gauge Dressing Occlusive",
226137: "18 Gauge placed in outside facility",
225184: "Eye Care",
225087: "Visual / hearing deficit",
225113: "Currently experiencing pain",
225126: "Dialysis patient",
224639: "Daily Weight",
50971: "Potassium",
50902: "Chloride",
50983: "Sodium",
50912: "Creatinine",
51006: "Urea Nitrogen",
50882: "Bicarbonate",
50868: "Anion Gap",
50931: "Glucose",
51221: "Hematocrit",
51265: "Platelet Count",
51301: "White Blood Cells",
51222: "Hemoglobin",
51279: "Red Blood Cells",
51250: "MCV",
51248: "MCH",
51249: "MCHC",
51277: "RDW",
50960: "Magnesium",
50970: "Phosphate",
50893: "Calcium, Total",
51274: "PT",
51237: "INR(PT)",
51275: "PTT-LAB",
50820: "pH",
50813: "Lactate",
50802: "Base Excess",
50821: "pO2",
50818: "pCO2",
50804: "Calculated Total CO2"
}

harutyunyan_list = [223951, 224308,
                    220051, 220180, 224643, 225310, 227242,
                    223835,
                    220739,
                    223900,
                    223901,
                    220621, 225664, 226537, 228388,
                    220045,
                    226707, 226730,
                    220052, 220181,
                    220227, 220277,
                    220210, 223851, 224689, 224690,
                    220050, 220179, 224167, 225309, 227243,
                    223761, 223762, 224027,
                    224639, 226512, 226531,
                    220274, 220734, 223830, 228243]
def read_itemid_to_variable_map(fn, variable_column='LEVEL2'):
    var_map = dataframe_from_csv(fn, index_col=None).fillna('').astype(str)
    # var_map[variable_column] = var_map[variable_column].apply(lambda s: s.lower())
    var_map.COUNT = var_map.COUNT.astype(int)
    var_map.ITEMID = var_map.ITEMID.astype(int)
    var_map = var_map[(var_map.ITEMID.isin(harutyunyan_list)) | (var_map.ITEMID.isin(additional_vars.keys()))]
    #print('additional_vars key len=', len(additional_vars.keys()))
    #print('no columns covered=', var_map.shape[0])

    #var_map = var_map[(var_map.STATUS == 'ready')]
    var_map = var_map[[variable_column, 'ITEMID', 'MIMIC LABEL']].set_index('ITEMID')
    var_map.rename({variable_column: 'VARIABLE', 'MIMIC LABEL': 'MIMIC_LABEL'}, axis=1, inplace=True)
    #print(var_map)
    additional_df = DataFrame(list(additional_vars.items()), columns=['ITEMID', 'VARIABLE']).set_index('ITEMID')

    var_map.update(additional_df)

    additional_vars_keys = set(additional_vars.keys())
    var_map_itemids = set(var_map.index.unique())
    keys_not_found = additional_vars_keys - var_map_itemids

    for key in keys_not_found:
        var_map = var_map.append(DataFrame({'VARIABLE': additional_vars[key], 'MIMIC_LABEL': additional_vars[key]}, index=[key]))

    var_map['VARIABLE'] = var_map['VARIABLE'].replace('', NA)
    var_map['VARIABLE'].fillna(var_map['MIMIC_LABEL'], inplace=True)

    # Ensure VARIABLE values are unique; if not, replace them with MIMIC_LABEL
    duplicate_vars = var_map[var_map.duplicated(['VARIABLE'], keep=False)]
    for idx in duplicate_vars.index:
        var_map.at[idx, 'VARIABLE'] = var_map.at[idx, 'MIMIC_LABEL']
        additional_vars[idx] = var_map.at[idx, 'VARIABLE']  # Store for future use

    #set_option('display.max_rows', None)
    #print(var_map)
    #print('final no columns covered=', var_map.shape[0])
    #print('unique variables=', len(var_map['VARIABLE'].unique()))
    #exit(1)

    return var_map


def map_itemids_to_variables(events, var_map):
    return events.merge(var_map, left_on='ITEMID', right_index=True)


def read_variable_ranges(fn, variable_column='LEVEL2'):
    columns = [variable_column, 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH']
    to_rename = dict(zip(columns, [c.replace(' ', '_') for c in columns]))
    to_rename[variable_column] = 'VARIABLE'
    var_ranges = dataframe_from_csv(fn, index_col=None)
    # var_ranges = var_ranges[variable_column].apply(lambda s: s.lower())
    var_ranges = var_ranges[columns]
    var_ranges.rename(to_rename, axis=1, inplace=True)
    var_ranges = var_ranges.drop_duplicates(subset='VARIABLE', keep='first')
    var_ranges.set_index('VARIABLE', inplace=True)
    return var_ranges.loc[var_ranges.notnull().all(axis=1)]


def remove_outliers_for_variable(events, variable, ranges):
    if variable not in ranges.index:
        return events
    idx = (events.VARIABLE == variable)
    v = events.VALUE[idx].copy()
    v.loc[v < ranges.OUTLIER_LOW[variable]] = np.nan
    v.loc[v > ranges.OUTLIER_HIGH[variable]] = np.nan
    v.loc[v < ranges.VALID_LOW[variable]] = ranges.VALID_LOW[variable]
    v.loc[v > ranges.VALID_HIGH[variable]] = ranges.VALID_HIGH[variable]
    events.loc[idx, 'VALUE'] = v
    return events


# SBP: some are strings of type SBP/DBP
def clean_sbp(df):
    v = df.VALUE.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(1))
    return v.astype(float)


def clean_dbp(df):
    v = df.VALUE.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(2))
    return v.astype(float)


# CRR: strings with brisk, <3 normal, delayed, or >3 abnormal
def clean_crr(df):
    v = Series(np.zeros(df.shape[0]), index=df.index)
    v[:] = np.nan

    # when df.VALUE is empty, dtype can be float and comparision with string
    # raises an exception, to fix this we change dtype to str
    df_value_str = df.VALUE.astype(str)

    v.loc[(df_value_str == 'Normal <3 Seconds') | (df_value_str == 'Brisk')] = 0
    v.loc[(df_value_str == 'Abnormal >3 Seconds') | (df_value_str == 'Delayed')] = 1
    return v


# FIO2: many 0s, some 0<x<0.2 or 1<x<20
def clean_fio2(df):
    v = df.VALUE.astype(float).copy()

    ''' The line below is the correct way of doing the cleaning, since we will not compare 'str' to 'float'.
    If we use that line it will create mismatches from the data of the paper in ~50 ICU stays.
    The next releases of the benchmark should use this line.
    '''
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (v>1.0)

    ''' The line below was used to create the benchmark dataset that the paper used. Note this line will not work
    in python 3, since it may try to compare 'str' to 'float'.
    '''
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (df.VALUE > 1.0)

    ''' The two following lines implement the code that was used to create the benchmark dataset that the paper used.
    This works with both python 2 and python 3.
    '''
    is_str = np.array(map(lambda x: type(x) == str, list(df.VALUE)), dtype=np.bool)
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (is_str | (~is_str & (v > 1.0)))

    v.loc[idx] = v[idx] / 100.
    return v


# GLUCOSE, PH: sometimes have ERROR as value
def clean_lab(df):
    v = df.VALUE.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan
    return v.astype(float)


# O2SAT: small number of 0<x<=1 that should be mapped to 0-100 scale
def clean_o2sat(df):
    # change "ERROR" to NaN
    v = df.VALUE.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan

    v = v.astype(float)
    idx = (v <= 1)
    v.loc[idx] = v[idx] * 100.
    return v


# Temperature: map Farenheit to Celsius, some ambiguous 50<x<80
def clean_temperature(df):
    v = df.VALUE.astype(float).copy()
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'F' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'F' in s.lower()) | (v >= 79)
    v.loc[idx] = (v[idx] - 32) * 5. / 9
    return v


# Weight: some really light/heavy adults: <50 lb, >450 lb, ambiguous oz/lb
# Children are tough for height, weight
def clean_weight(df):
    v = df.VALUE.astype(float).copy()
    # ounces
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'oz' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'oz' in s.lower())
    v.loc[idx] = v[idx] / 16.
    # pounds
    idx = idx | df.VALUEUOM.fillna('').apply(lambda s: 'lb' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'lb' in s.lower())
    v.loc[idx] = v[idx] * 0.453592
    return v

def clean_dumb(df):
    return df.VALUE.copy()

# Height: some really short/tall adults: <2 ft, >7 ft)
# Children are tough for height, weight
def clean_height(df):
    v = df.VALUE.astype(float).copy()
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'in' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'in' in s.lower())
    v.loc[idx] = np.round(v[idx] * 2.54)
    return v


# ETCO2: haven't found yet
# Urine output: ambiguous units (raw ccs, ccs/kg/hr, 24-hr, etc.)
# Tidal volume: tried to substitute for ETCO2 but units are ambiguous
# Glascow coma scale eye opening
# Glascow coma scale motor response
# Glascow coma scale total
# Glascow coma scale verbal response
# Heart Rate
# Respiratory rate
# Mean blood pressure
clean_fns = {
    #'Capillary refill rate': clean_crr,
    'Capillary Refill R': clean_crr,
    'Capillary Refill L': clean_crr,
    #'Diastolic blood pressure': clean_dbp,
    'Arterial Blood Pressure diastolic': clean_dbp,
    'Non Invasive Blood Pressure diastolic': clean_dbp,
    'Manual Blood Pressure Diastolic Left': clean_dbp,
    'ART BP Diastolic': clean_dbp,
    'Manual Blood Pressure Diastolic Right': clean_dbp,
    #'Systolic blood pressure': clean_sbp,
    'Arterial Blood Pressure systolic': clean_sbp,
    'Non Invasive Blood Pressure systolic': clean_sbp,
    'Manual Blood Pressure Systolic Left': clean_sbp,
    'ART BP Systolic': clean_sbp,
    'Manual Blood Pressure Systolic Right': clean_sbp,
    'Fraction inspired oxygen': clean_fio2,
    'Oxygen saturation': clean_o2sat,
    'O2 saturation pulseoxymetry': clean_o2sat,
    #'Glucose': clean_lab,
    'Glucose (serum)': clean_lab,
    'Glucose finger stick (range 70-100)': clean_lab,
    'Glucose (whole blood)': clean_lab,
    'Glucose (whole blood) (soft)': clean_lab,
    'GLUCOSE': clean_lab,
    #'pH': clean_lab,
    'PH': clean_lab,
    'PH (Arterial)': clean_lab,
    'PH (SOFT)': clean_lab,
    'PH (Venous)': clean_lab,
    'PH (dipstick)': clean_lab,
    'Temperature': clean_temperature,
    'Temperature Fahrenheit': clean_temperature,
    'Skin Temperature': clean_dumb,
    'Weight': clean_weight,
    'Daily Weight': clean_weight,
    'Admission Weight (lbs.)': clean_weight,
    'Height': clean_height,
    'Height (cm)': clean_height,
    'Mean blood pressure': clean_dumb,
    'Rate': clean_dumb,
    'Respiratory Rate (Total)': clean_dumb,
    'Respiratory Rate (spontaneous)': clean_dumb,
}

def cleanup_for_duett(df):
    categorical_mappings = {
        "Braden Activity": {"Bedfast": 1, "Chairfast": 2, "Walks Occasionally": 3, "Walks Frequently": 4},
        "Braden Friction/Shear": {"Potential Problem": 1, "No Apparent Problem": 2, "Problem": 3},
        "Braden Mobility": {"No Limitations": 1, "Slight Limitations": 2, "Very Limited": 3, "Completely Immobile": 4},
        "Braden Moisture": {"Rarely Moist": 1, "Occasionally Moist": 2, "Moist": 3, "Consistently Moist": 4},
        "Braden Nutrition": {"Excellent": 1, "Adequate": 2, "Probably Inadequate": 3, "Very Poor": 4},
        "Braden Sensory Perception": {"No Impairment": 1, "Slight Impairment": 2, "Very Limited": 3,
                                      "Completely Limited": 4},
        "GCS - Eye Opening": {"Spontaneously": 4, "To Speech": 3, "To Pain": 2, "None": 1},
        "GCS - Motor Response": {"Obeys Commands": 6, "Localizes Pain": 5, "Flex-withdraws": 4, "Abnormal Flexion": 3,
                                 "Abnormal extension": 2, "No response": 1},
        "GCS - Verbal Response": {"Oriented": 5, "Confused": 4, "Inappropriate Words": 3, "Incomprehensible sounds": 2,
                                  "No Response": 1, "No Response-ETT": 0},
        "Goal Richmond-RAS Scale": {" 0  Alert and calm": 0, "-1 Awakens to voice (eye opening/contact) > 10 sec": -1,
                                    "-2 Light sedation, briefly awakens to voice (eye opening/contact) < 10 sec": -2,
                                    "-3 Moderate sedation, movement or eye opening; No eye contact": -3,
                                    "-4 Deep sedation, no response to voice, but movement or eye opening to physical stimulation": -4,
                                    "-5 Unarousable, no response to voice or physical stimulation": -5},
        "Pain Level": {"None": 0, "Mild": 1, "Mild to Moderate": 2, "Moderate": 3, "Moderate to Severe": 4, "Severe": 5, "Severe to Worse": 6, "Worst": 7, "Unable to Score": 8},
        "Pain Level Response": {"None": 0, "Mild": 1, "Mild to Moderate": 2, "Moderate": 3, "Moderate to Severe": 4, "Severe": 5, "Severe to Worse": 6, "Worst": 7, "Unable to Score": 8},
        "Richmond-RAS Scale": {" 0  Alert and calm": 0, "+1 Anxious, apprehensive, but not aggressive": 1,
                               "+2 Frequent nonpurposeful movement, fights ventilator": 2,
                               "+3 Pulls or removes tube(s) or catheter(s); aggressive": 3,
                               "+4 Combative, violent, danger to staff": 4,
                               "-1 Awakens to voice (eye opening/contact) > 10 sec": -1,
                               "-2 Light sedation, briefly awakens to voice (eye opening/contact) < 10 sec": -2,
                               "-3 Moderate sedation, movement or eye opening; No eye contact": -3,
                               "-4 Deep sedation, no response to voice, but movement or eye opening to physical stimulation": -4,
                               "-5 Unarousable, no response to voice or physical stimulation": -5},
        "Strength L Arm": {"No movement": 0, "Muscle contraction, but no movement": 1,
                           "Movement, but not against gravity": 2, "Lifts against gravity, no resistance": 3,
                           "Some resistance": 4, "Full resistance": 5,
                           },
        "Strength L Leg": {"No movement": 0, "Muscle contraction, but no movement": 1,
                           "Movement, but not against gravity": 2, "Lifts against gravity, no resistance": 3,
                           "Some resistance": 4, "Full resistance": 5,
                           },
        "Strength R Arm": {"No movement": 0, "Muscle contraction, but no movement": 1,
                           "Movement, but not against gravity": 2, "Lifts against gravity, no resistance": 3,
                           "Some resistance": 4, "Full resistance": 5,
                           },
        "Strength R Leg": {"No movement": 0, "Muscle contraction, but no movement": 1,
                           "Movement, but not against gravity": 2, "Lifts against gravity, no resistance": 3,
                           "Some resistance": 4, "Full resistance": 5,
                           },
        "Ambulatory aid": {"None": 0, "Cane": 1, "Crutches": 2, "Walker": 3, "Furniture": 4,
                           "Wheel chair": 5, "Nurse assist": 6, "Bed rest": 7,
                           },
        #"Capillary Refill L": {"Normal <3 Seconds": 1, "Abnormal >3 Seconds": 2},
        #"Capillary Refill R": {"Normal <3 Seconds": 1, "Abnormal >3 Seconds": 2},
        "Gait/Transferring": {"Normal ": 1, "Impaired": 2, "Weak": 3, "Bed rest": 4, "Immobile": 5},
        "History of falling (within 3 mnths)": {"Yes": 1, "No": 0},
        "IV/Saline lock": {"Yes": 1, "No": 0},
        "Mental status": {"Oriented to own ability": 1, "Forgets limitations": 2},
        "Secondary diagnosis": {"Yes": 1, "No": 0},
        "Skin Temperature": {"Cool": 1, "Warm": 2, "Hot": 3}
    }

    # print(df.columns)
    # Apply categorical mappings
    for column, mapping in categorical_mappings.items():
        df[column] = df[column].map(mapping)

    # Numeric columns to convert
    #numeric_columns = ['PTT-LAB', 'PTT-CHART', 'pCO2', 'pO2', 'Hemoglobin', 'Magnesium', 'Lactate', 'Calculated Total CO2',
    #                   'Potassium', 'Anion Gap', 'Sodium', 'Bicarbonate', 'Hematocrit', 'Platelet Count',
    #                   'White Blood Cells', 'Calcium, Total', 'Urea Nitrogen', 'MCV', 'Phosphate', 'Creatinine',
    #                   'INR(PT)', 'Chloride', 'PT', 'MCHC', 'MCH', 'Red Blood Cells', 'Base Excess', 'RDW']

    # Convert numeric columns and handle special text entries like "___", "ERROR", or "UNABLE TO REPORT" by setting them to NaN
    for column in df.columns:
        df[column] = to_numeric(df[column], errors='coerce')

    return df

def clean_events(events):
    global clean_fns
    additional_var_names = list(additional_vars.values())
    for var_name, clean_fn in clean_fns.items():
        idx = (events.VARIABLE == var_name)
        try:
            events.loc[idx, 'VALUE'] = clean_fn(events[idx])
        except Exception as e:
            import traceback
            print("Exception in clean_events:", clean_fn.__name__, e)
            print(traceback.format_exc())
            print("number of rows:", np.sum(idx))
            print("values:", events[idx])
            exit()
    cleaned_or_additional_vars = set(clean_fns.keys()).union(additional_var_names)
    events = events.loc[events['VARIABLE'].isin(cleaned_or_additional_vars) & events['VALUE'].notnull()]

    return events
