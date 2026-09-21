# stellar-class-prediction

Classifying astronomical objects as GALAXY, QSO or STAR — [Kaggle Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6).

## Result

| | |
|---|---|
| Final rank | 950 / 2816 |
| Best public score | 0.96663 |
| Best CV | 0.9650 |
| Leader | 0.97289 |

## Dataset

- 577,347 training rows
- **Numerical:** alpha, delta, u, g, r, i, z, redshift
- **Categorical:** spectral_type (A/F, G/K, O/B, M), galaxy_population (Red_Sequence, Blue_Cloud)
- **Target:** GALAXY ~370k · QSO ~120k · STAR ~80k (approximate)
- **Metric:** balanced accuracy

## Approach

- One-hot encoded `spectral_type`, mapped `galaxy_population` to 0/1, label-encoded the target
- Engineered colour indices from adjacent brightness bands: u−g, g−r, r−i, i−z
- Added log(1 + redshift) as a new feature
- LightGBM with `class_weight='balanced'`, evaluated with 5-fold StratifiedKFold
- Tuned with Optuna, including L1/L2 regularisation (`reg_alpha`, `reg_lambda`)
- Tested CatBoost and XGBoost ensembles, searching blend weights on out-of-fold predictions

## What worked / what didn't

| Step | CV | Public score |
|---|---|---|
| Default LightGBM | 0.9508 | 0.95053 |
| + colour indices and log redshift | 0.9515 | 0.95209 |
| Optuna, 25 trials | 0.9637 | 0.96560 |
| Optuna, 50 trials, wider ranges | — | 0.96647 |
| Optuna, 58 trials + L1/L2 | — | 0.96663 |
| + tuned CatBoost ensemble | 0.9650 | 0.96623 |
| PyTorch neural network | — | 0.89691 |

Added nothing: ratio features, clipping colour-index outliers, and an XGBoost ensemble.

## Key takeaways

- Hyperparameter tuning was the biggest lever: +0.012 CV over the default model
- Engineered features helped only slightly: +0.0007 CV
- Ensembling other gradient boosters gave no real gain — LightGBM took 0.95 of the blend weight every time, as the models made correlated errors
- A minimal, untuned neural network scored 0.897, far below tuned trees — consistent with tree models generally leading on tabular data

## Files

- `stellar_class_prediction.ipynb` — full pipeline
- `requirements.txt` — dependencies
- Data: download from the competition page
