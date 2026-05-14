# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ML project predicting Spotify streaming performance (log-transformed stream counts) from audio features and platform exposure metrics. Single Jupyter notebook project — no build system, no test suite.

## Running the Notebook

```bash
jupyter notebook spotify_proj_code.ipynb
```

The dataset must be placed at `data/Popular_Spotify_Songs.csv` before running. Download from [Kaggle](https://www.kaggle.com/datasets/abdulszz/spotify-most-streamed-songs).

## Environment

```bash
pip install -r requirements.txt
```

Project was developed in Google Colab and Anaconda. The notebook uses a local `data/` path — the Colab version uses a Google Drive mount path instead.

## Notebook Structure

The notebook (`spotify_proj_code.ipynb`) follows this flow:

1. **Data loading and cleaning** — fix string-encoded numeric columns, fill missing values, cast types
2. **Feature engineering** — log-transform target (`log_streams = log1p(streams)`), define numeric and categorical feature sets
3. **Preprocessing pipeline** — `ColumnTransformer` with `StandardScaler` for numeric features and `OneHotEncoder` for categorical (`key`, `mode`)
4. **Linear models** — Linear Regression, Ridge, Lasso, Elastic Net using a reduced feature set (`linear_numeric_features`) after VIF analysis removed `released_year` (VIF ~100) and high-collinearity audio features
5. **Nonlinear models** — Random Forest, SVR, XGBoost using the full feature set (`numeric_features` + `categorical_features`)
6. **Hyperparameter tuning** — GridSearchCV with 3-fold CV on Random Forest; best params: `max_depth=15`, `min_samples_split=10`, `n_estimators=200`
7. **Evaluation** — RMSE, MAE, R² on 80/20 held-out test set; Random Forest is best (R²=0.736)
8. **Error analysis** — residual plot, per-artist MAE breakdown for 16 major artists

## Key Variables

- `numeric_features` — full feature list used by nonlinear models (note: `released_year` is dropped mid-notebook via `df.drop`)
- `linear_numeric_features` — reduced feature list for linear models (excludes high-VIF features)
- `nonlinear_preprocessor` — alias for the original `preprocessor` ColumnTransformer
- `best_rf` — the tuned Random Forest pipeline (result of GridSearchCV)
- `X_train`, `X_test`, `y_train`, `y_test` — 80/20 split; `X_test_raw` is the un-preprocessed test set used for artist error analysis
