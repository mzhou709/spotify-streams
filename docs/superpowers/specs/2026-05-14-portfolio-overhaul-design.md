# Design: Spotify Streams — Portfolio Overhaul

**Date:** 2026-05-14
**Goal:** Transform a school ML project into a professional portfolio piece targeting DS and MLE roles.
**Scope:** One weekend. Notebook narrative cleanup, Streamlit prediction app, deployment, README refresh.

---

## Context

The project predicts log-transformed Spotify stream counts from audio features and platform exposure metrics using a tuned Random Forest (R2 = 0.736). The code is solid but the repo presents as a school assignment: no markdown narrative, duplicate exploratory cells, a formal PDF report linked from the README, and a Colab badge no recruiter will click. The goal is to reframe the same work as a deliberate personal project with a live demo.

---

## File Structure (after)

```
spotify-streams/
├── analysis.ipynb            # renamed from spotify_proj_code.ipynb
├── app.py                    # new — Streamlit prediction demo
├── model/
│   └── rf_pipeline.pkl       # new — saved trained pipeline (joblib)
├── data/
│   └── .gitkeep
├── requirements.txt          # updated — add streamlit, joblib
└── README.md                 # updated — demo link, leaner structure
```

`spotify_proj.pdf` is deleted from the repo entirely.

---

## 1. Notebook Restructure (analysis.ipynb)

### What gets removed
- **Cells 8-15:** Pre-VIF exploratory model runs (LR, RF, SVR, Decision Tree, XGBoost, CV RMSE, Ridge, Lasso). These are scratch work — the same models get fit properly after VIF cleanup. Removing them eliminates redundant output and makes the analysis read as intentional.

### Markdown narrative — 8 sections

Markdown cells explain decisions and reasoning only — never what the code does, never concepts a DS/MLE interviewer already knows.

| Section | Narrative focus |
|---|---|
| 1. Problem and Dataset | Predicting streaming success is hard because it is driven by both intrinsic audio qualities and platform distribution — this dataset lets us separate those signals. Why log1p: raw streams are heavily right-skewed; log-transforming stabilizes variance and keeps linear model assumptions closer to valid. |
| 2. Data Cleaning | Several numeric columns (streams, in_deezer_playlists, in_shazam_charts) were string-encoded with commas and dashes in the source CSV — cleaned to numeric. key nulls filled as "Unknown" to preserve rows rather than drop ~10% of the dataset. |
| 3. Feature Engineering | log_streams = log1p(streams) is the target. Two feature sets are defined here for a reason covered in the next section. |
| 4. Multicollinearity (VIF) | VIF run before fitting linear models — released_year (VIF ~100) and correlated audio features inflate standard errors and distort linear coefficients. Two feature sets: linear_numeric_features drops high-VIF features; nonlinear models use the full set since tree-based methods handle collinearity without issue. |
| 5. Modeling Approach | Linear models on the reduced feature set establish an interpretable baseline. Nonlinear models test whether feature interactions improve prediction — expected given that playlist placement likely compounds with audio appeal in non-additive ways. |
| 6. Model Comparison | Platform exposure features dominate in both linear coefficients and RF feature importances. Nonlinear models substantially outperform linear (RF R2 0.736 vs ~0.48), confirming interaction effects linear models cannot capture. |
| 7. Hyperparameter Tuning | GridSearchCV with 3-fold CV on RF. Best: max_depth=15, min_samples_split=10, n_estimators=200 — shallow enough to avoid overfitting on 953 rows. |
| 8. Error Analysis | Residual plot shows systematic underprediction at the high end — the model cannot see TikTok virality, film/TV placements, or radio rotation. Artist-level MAE: Feid and Morgan Wallen predicted well (catalog-driven); Ed Sheeran and Kendrick Lamar have the highest error (event-driven spikes the features cannot capture). |

### Model export cell (added at end)

```python
import joblib, os
os.makedirs("model", exist_ok=True)
joblib.dump(best_rf, "model/rf_pipeline.pkl")
```

---

## 2. Streamlit App (app.py)

### Purpose
Let a visitor input song/platform characteristics and see a predicted stream count — no notebook required. Demonstrates end-to-end model serving.

### Layout
- **Left sidebar:** Input controls
- **Main panel:** Prediction output

### Inputs (sidebar)

Platform exposure (top predictors):
- in_spotify_playlists — slider 0 to 10,000
- in_apple_playlists — slider 0 to 5,000
- in_deezer_playlists — slider 0 to 5,000
- in_spotify_charts — slider 0 to 200

Audio features (sliders 0-100):
- danceability_%, energy_%, valence_%

Other:
- bpm — slider 60 to 200
- artist_count — slider 1 to 4
- key — dropdown (C, C#, D, D#, E, F, F#, G, G#, A, A#, B, Unknown)
- mode — dropdown (Major, Minor)

Remaining features default to training dataset medians silently (low importance, would clutter UI).

### Main panel output
1. Predicted streams — back-transformed via expm1, formatted with commas
2. Percentile context — where the prediction falls in training set distribution ("Top X% of songs in the dataset")
3. One-line interpretation — bucketed label based on percentile (e.g. "Mid-tier chart hit", "Niche/catalog track", "Potential viral hit")

### Data flow

```
user inputs → build input dict → fill medians for hidden features
→ pd.DataFrame (1 row, all expected columns)
→ best_rf.predict() → log-prediction → np.expm1() → display
```

---

## 3. README

Lean structure optimized for a 30-second recruiter scan. No "Tech Stack" section — listing pandas/sklearn adds no signal and reads as filler. Libraries are self-evident from requirements.txt. No "Overview" header — the description IS the overview.

```markdown
# Spotify Streams Prediction
[Live Demo badge]

One-line description: what you predicted + best result (R2 = 0.736).

## Key Findings
- Platform exposure dominates: playlist counts across Spotify, Apple, Deezer outweigh every audio feature
- Nonlinear models win: RF R2 0.736 vs linear ~0.48 — streaming success has interaction effects linear models miss
- Multicollinearity in audio features: VIF analysis flagged released_year (VIF ~100) and correlated audio features; removed from linear models, kept for tree-based
- Viral hits resist prediction: systematic underprediction for Ed Sheeran, Kendrick Lamar — TikTok virality and sync placements are invisible to the model

## Model Comparison
| table |

## Setup
pip install -r requirements.txt
streamlit run app.py
# or: jupyter notebook analysis.ipynb

## Data
Kaggle link — place Popular_Spotify_Songs.csv in data/
```

---

## 4. Deployment

Streamlit Community Cloud (free):
1. Push repo to GitHub (already there)
2. share.streamlit.io — connect repo, set app.py as entry point
3. Copy assigned URL into README badge

rf_pipeline.pkl committed to repo (a few MB for 200 trees — fine for GitHub).

---

## Out of Scope (this weekend)

- SHAP interpretability — good stretch goal once recruiting is underway
- src/ package structure, Docker

---

## Success Criteria

- Recruiter can click README demo link and interact with the model in under 10 seconds
- Notebook reads as deliberate analytical narrative, no redundant cells, no textbook explanations
- No school-project artifacts in repo (PDF gone, old notebook name gone)
- requirements.txt installs cleanly and streamlit run app.py works locally
