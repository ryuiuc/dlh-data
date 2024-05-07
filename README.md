MIMIC-IV 2.0 Data Processing for DuETT Paper
=========================
This repository contains the data processing modules developed for replicating the DuETT paper's methodologies using the MIMIC-IV 2.0 dataset. The DuETT paper's analysis is focused on the intensive care unit (ICU) mortality predictions.

DuETT: Dual Event Time Transformer for Electronic Health Records

DuETT Github: https://github.com/layer6ai-labs/duett

DuETT Paper: https://www.cs.toronto.edu/~mvolkovs/MLH2023_DuETT.pdf https://arxiv.org/abs/2304.13017

## Overview
The project builds upon the foundational work by Harutyunyan et al. (2019), which originally developed benchmarks for the MIMIC-III dataset. This repository extends and adapts these benchmarks for the newer MIMIC-IV 2.0 dataset, following the methods detailed in the DuETT paper.

## Motivation
The primary goal of this repository is to replicate the data preprocessing and analysis performed in the DuETT paper, but with updates to accommodate the latest version of the MIMIC dataset:

1.  Dataset Upgrade: Transition from MIMIC-III to MIMIC-IV 2.0 to utilize more recent and comprehensive clinical data.
2.  Customization: Implement customized data preprocessing methods as specified in the DuETT paper, tailored specifically for ICU mortality prediction task.

## Source
This work is based on an upgraded data processing pipeline initially created for MIMIC-IV 1.0, available at https://github.com/vincenzorusso12/mimic4-benchmarks. Our adaptations further refine these processes to align with the specific analyses and methodologies outlined in the DuETT paper.

## Limitations
Currently, this repository is focused exclusively on the end-to-end data pipeline for ICU mortality. Future updates may expand to cover additional metrics and outcomes based on further analyses and community contributions.

## Building data for in-hospital mortality task

Here are the required steps to build the benchmark. It assumes that you already have MIMIC-IV dataset (lots of CSV files) on the disk. 

1. Clone the repo.

       git clone https://github.com/ryuiuc/dlh-data/
       cd mimic4-benchmarks/
    
2. The following command takes MIMIC-IV CSVs, generates one directory per `SUBJECT_ID` and writes ICU stay information to `data/{SUBJECT_ID}/stays.csv`, diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`. This step might take around an hour.

       python -m mimic4benchmark.scripts.extract_subjects ./mimic-iv data/root/

3. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows (more information can be found in [`mimic4benchmark/scripts/more_on_validating_events.md`](mimic4benchmark/scripts/more_on_validating_events.md)).

       python -m mimic4benchmark.scripts.validate_events data/root/

4. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```{SUBJECT_ID}/episode{#}_timeseries.csv``` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in ```{SUBJECT_ID}/episode{#}.csv```. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). **Outlier detection is disabled in the current version**.

       python -m mimic4benchmark.scripts.extract_episodes_from_subjects data/root/

5. The next command splits the whole dataset into training and testing sets. Note that the train/test split is the same of all tasks.

       python -m mimic4benchmark.scripts.split_train_and_test data/root/
	
6. The following commands will generate task-specific datasets, which can later be used in models. These commands are independent, if you are going to work only on one benchmark task, you can run only the corresponding command.

       python -m mimic4benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/

After the above commands are done, there will be a directory `data/{task}` for in-hospital mortality task.
These directories have two sub-directories: `train` and `test`.
Each of them contains bunch of ICU stays and one file with name `listfile.csv`, which lists all samples in that particular set.

NOTE: To reproduce the DuETT results, use below to make sure deterministic results:
- `resources/testset.csv` is derived from `resources/mimic-iv-patient-split.json` of the DuETT paper.
- `resources/train-listfile.csv` is the train data split corresponding to `data/in-hospital-mortality/train/listfile.csv`.
- `resources/test-listfile.csv` is the test data split corresponding to `data/in-hospital-mortality/test/listfile.csv`.


## Readers
Ignore. Not used.


## Evaluation
Ignore. Not used.


## Other Models
Ignore. Not used.


