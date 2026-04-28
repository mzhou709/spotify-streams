# Spotify Streams Prediction 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mzhou709/spotify-streams/blob/main/spotify_proj_code.ipynb) 

Predicting Spotify streaming performance from audio features and platform exposure metrics across 953 tracks. A tuned Random Forest achieved the best out-of-sample performance (RMSE 0.516, MAE 0.403, R² 0.736).

## Overview

- **Goal:** Predict log-transformed stream counts using audio characteristics (danceability, energy, BPM, valence, etc.) and platform exposure metrics (playlist and chart appearances across Spotify, Apple Music, Deezer, and Shazam)
- **Dataset:** [Most Streamed Spotify Songs 2023](https://www.kaggle.com/datasets/abdulszz/spotify-most-streamed-songs) — 953 tracks, 25 features
- **Target:** `log(1 + streams)` — log-transformed to compress right skew and improve model stability

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, statsmodels, matplotlib, seaborn

## Models Compared

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest (tuned) | 0.516 | 0.403 | 0.736 |
| SVR | 0.663 | 0.529 | 0.565 |
| Elastic Net | 0.719 | 0.589 | 0.489 |
| Linear Regression | 0.723 | 0.577 | 0.483 |

## Key Findings

- **Platform exposure dominates:** Spotify, Deezer, and Apple playlist counts were the top predictors in both linear coefficients and Random Forest feature importances — songs on more playlists stream significantly more
- **Nonlinear models win:** Random Forest substantially outperformed all linear models (R² 0.736 vs ~0.49), indicating that streaming success depends on feature interactions that linear models cannot capture (e.g., high playlist count combined with high danceability compounding engagement)
- **Multicollinearity in audio features:** VIF analysis revealed high collinearity among `released_year` (VIF ~100), `danceability_%`, and `energy_%`, leading to a reduced feature set for linear models
- **Viral hits are hard to predict:** All models consistently underpredicted the highest-stream songs — external signals like TikTok virality, film/TV placements, and radio play are absent from the dataset. Ed Sheeran, Kendrick Lamar, and Labrinth had the highest per-artist prediction error

## Data

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/abdulszz/spotify-most-streamed-songs) and place the CSV in the `data/` folder.

## Full report
[📄 Report](spotify_proj.pdf)
