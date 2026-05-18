# Spotify Streams Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://spotify-streams-predictor.streamlit.app/)

Predicting Spotify streaming performance (log-transformed stream counts) from audio features and platform exposure metrics across 953 tracks. Best model: tuned Random Forest (R² = 0.736).

## Key Findings

- **Platform exposure dominates:** Spotify, Apple, and Deezer playlist counts were the top predictors by a wide margin — songs on more playlists stream significantly more regardless of audio characteristics
- **Nonlinear models win:** Random Forest (R² 0.736) substantially outperformed all linear models (~0.48) — streaming success depends on feature interactions linear models can't capture
- **Multicollinearity in audio features:** VIF analysis flagged `released_year` (VIF ~100) and several correlated audio features; dropped from linear models, kept for tree-based
- **Viral hits resist prediction:** Systematic underprediction for Ed Sheeran, Kendrick Lamar, and Labrinth — TikTok virality, sync placements, and radio play are absent from the dataset

## Model Comparison

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest (tuned) | 0.516 | 0.403 | 0.736 |
| SVR | 0.663 | 0.529 | 0.565 |
| Elastic Net | 0.719 | 0.589 | 0.489 |
| Linear Regression | 0.723 | 0.577 | 0.483 |

## Project Structure

| File | Description |
|---|---|
| `analysis.ipynb` | Full analysis: cleaning, VIF, modeling, tuning, error analysis |
| `app.py` | Streamlit prediction demo |
| `model/rf_pipeline.pkl` | Saved Random Forest pipeline (run notebook to generate) |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
# or: jupyter notebook analysis.ipynb
```

## Data

Download [Most Streamed Spotify Songs 2023](https://www.kaggle.com/datasets/abdulszz/spotify-most-streamed-songs) from Kaggle and place `Popular_Spotify_Songs.csv` in the `data/` folder.
