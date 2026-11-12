# Indoor air quality classification and decision support (CNN–LSTM–TCN)

Code and results for the paper *"A Hybrid CNN–LSTM–TCN Deep Learning Framework for Indoor Air Quality Monitoring and Decision-Making in Classrooms"*.

The pipeline turns raw classroom sensor readings (CO2, PM2.5, PM10, temperature, relative humidity) into six IAQI-S classes, compares nine architectures on rooms not seen in training, and maps the predicted class to an alert level.

## Contents

All files are in the main folder.

| File | What it holds |
|---|---|
| `01_pipeline_models_external_validation.ipynb` | Resampling, cleaning, IAQI-S labels, room-level split, windowing, oversampling, the nine models (multi-seed), ablation, external validation on the Bangladeshi classroom data |
| `02_analysis_decision_layer_alerts.ipynb` | EVT/GPD, Markov chain, SHAP, rule-based decision layer, dashboard sample, Firebase alert test |
| `kaggle_data_prep_and_ablation.py` | Data preparation and ablation as run on Kaggle (fixed seed 42) |
| `room_assignment_table.py` | Prints the room assignment and per-class window counts of each set |
| `results_*.csv`, `nine_models_summary.csv` | Result tables used in the paper |

Outputs of the notebooks were cleared. Run them to regenerate the figures and tables. The notebooks also contain a few exploratory runs that are not part of the paper.

## Data

The data are public and are not copied here.

- South African classroom dataset (Mendeley Data): https://data.mendeley.com/datasets/tys2gscdv7/11
- Bangladeshi classroom dataset (IUB Academic Repository): http://ar.iub.edu.bd/handle/11348/992
- DALTON indoor dataset: https://github.com/prasenjit52282/dalton-dataset

## Environment

- Python 3.13, TensorFlow 2.21, NumPy 2.1, pandas 2.3, scikit-learn 1.6
- Kaggle notebook, NVIDIA Tesla T4 (15 GB), Intel Xeon CPU at 2.0 GHz
- Package list: `requirements.txt` (versions as in the environment above; the notebooks also use matplotlib, scipy, shap, joblib and firebase-admin)

## Reproducing the results

1. Download the datasets and set the paths at the top of the notebooks (the notebooks use Google Drive/Kaggle paths).
2. Run `01_...` from the first cell. The split is deterministic. The room assignment is given in the paper (Appendix, Table A1) and is printed by `room_assignment_table.py`.
3. Random seeds: 1, 7, 42, 123 and 2024. Model selection and the per-class tables use seed 42; mean ± standard deviation values use all five seeds.
4. Run `02_...` for the EVT, Markov, SHAP and decision-layer results.

## Firebase and alerts

The alert test needs your own Firebase project. The database URL and the device token were replaced by placeholders (`YOUR_FIREBASE_DATABASE_URL`, `YOUR_DEVICE_TOKEN`). Put your own service-account file at the path given in the notebook. Never commit these files.

## License and data terms

Add a license of your choice for the code (for example MIT). The DALTON data are released under their own license (AGPL, non-commercial research use).
