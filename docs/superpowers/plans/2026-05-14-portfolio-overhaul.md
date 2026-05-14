# Spotify Streams Portfolio Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform a school ML project into a professional DS/MLE portfolio piece with a live Streamlit demo.

**Architecture:** Clean up repo artifacts, restructure the notebook as a narrative analysis, build a Streamlit prediction app backed by a saved sklearn pipeline, and update the README to lead with a live demo link. The notebook restructure and app build are independent and run in parallel.

**Tech Stack:** Python, scikit-learn, joblib, streamlit, pandas, numpy; jupyter-notebook skill for notebook edits; developing-with-streamlit skill for app.py

---

## Task 1: Repo Cleanup (inline, ~5 min)

**Files:**
- Delete: `spotify_proj.pdf`
- Rename: `spotify_proj_code.ipynb` → `analysis.ipynb`
- Modify: `requirements.txt`
- Create: `model/.gitkeep`

- [ ] **Step 1: Delete the PDF**

```bash
git rm spotify_proj.pdf
```

- [ ] **Step 2: Rename the notebook**

```bash
git mv spotify_proj_code.ipynb analysis.ipynb
```

- [ ] **Step 3: Remove the ipynb checkpoint for the old name if present**

```bash
rm -f ".ipynb_checkpoints/spotify_proj_code-checkpoint.ipynb"
```

- [ ] **Step 4: Update requirements.txt**

Replace contents of `requirements.txt` with:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
statsmodels
jupyter
joblib
streamlit
```

- [ ] **Step 5: Create model directory**

```bash
mkdir -p model
touch model/.gitkeep
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: rename notebook, remove school report, add streamlit/joblib deps"
```

---

## Task 2 + 3: Parallel — Notebook Restructure AND Streamlit App

Dispatch both agents simultaneously. They are fully independent.

---

### Task 2: Notebook Restructure (parallel agent)

**Agent prompt:**

```
You are restructuring a Jupyter notebook at C:\Users\matth\proj\spotify-streams\analysis.ipynb
to read as a deliberate ML analysis rather than scratch work. Use the jupyter-notebook skill.

CONTEXT:
- This is a portfolio project for a new grad targeting DS/MLE roles
- The notebook predicts log-transformed Spotify stream counts using a tuned Random Forest (R2=0.736)
- Recruiters will read this notebook on GitHub — it must tell a clear story

WHAT TO REMOVE:
Delete cells 8 through 15 (0-indexed). These are exploratory first-pass model runs
(LinearRegression, RandomForest, SVR, DecisionTree, XGBoost, CV RMSE, Ridge, Lasso)
that happen BEFORE the VIF analysis. They are scratch work — the same models get fit
properly after VIF cleanup later in the notebook. Removing them makes the analysis
read as intentional rather than iterative trial-and-error.

WHAT TO ADD — 8 markdown cells inserted before their corresponding code sections:

1. Before the data loading cell (cell 1), insert:
## Predicting Spotify Streaming Performance

Streaming success is driven by two distinct forces: intrinsic audio qualities (tempo,
energy, danceability) and platform distribution (playlist placements, chart appearances).
This dataset lets us separate those signals and measure which actually predicts streams.

The target is log1p(streams) rather than raw streams — stream counts are heavily
right-skewed with a long tail of viral outliers. Log-transforming compresses that skew,
stabilizes variance, and keeps linear model residuals better behaved.

2. Before the data cleaning cell (cell 2), insert:
## Data Cleaning

Three columns — streams, in_deezer_playlists, in_shazam_charts — arrived as strings
with comma separators and occasional dashes. Converted to numeric; dashes treated as 0.
The key column had ~10% nulls, filled as "Unknown" to preserve those rows rather than
drop them.

3. Before the feature definition cell (cell 5), insert:
## Features

Two feature sets are defined here. The reason becomes clear after the VIF analysis below —
linear models need a reduced set to avoid collinearity problems, while tree-based methods
can use everything.

4. Before the preprocessor cell (cell 6), insert:
## Preprocessing Pipeline

StandardScaler on numeric features, OneHotEncoder on key and mode. Wrapped in a
ColumnTransformer so the same pipeline handles both linear and nonlinear models cleanly.

5. Before the VIF cell (cell 16), insert:
## Multicollinearity Analysis

Before fitting linear models, VIF (Variance Inflation Factor) identifies features whose
information is already captured by other features. High VIF inflates standard errors and
makes linear coefficients unreliable.

released_year came in at VIF ~100 — nearly perfectly collinear with other features.
danceability_% and energy_% also showed high collinearity. These are dropped from the
linear feature set. Tree-based models are unaffected by collinearity so they keep the
full feature set.

6. Before the linear/nonlinear model fitting cells, insert:
## Modeling

Linear models (Ridge, Lasso, ElasticNet) on the reduced feature set test whether
streaming success is linearly predictable from these signals. Nonlinear models (Random
Forest, SVR) test whether feature interactions improve prediction — playlist count
combined with high danceability may compound in ways a linear model cannot capture.

7. Before the GridSearchCV cell (cell 33), insert:
## Hyperparameter Tuning

GridSearchCV with 3-fold cross-validation over n_estimators, max_depth, and
min_samples_split. With only 953 rows, unconstrained trees overfit easily — the search
is designed to find the right depth constraint.

8. Before the residual plot cell (cell 35), insert:
## Error Analysis

The residual plot and per-artist MAE breakdown reveal where the model systematically
fails. Songs with the highest stream counts are consistently underpredicted — the model
has no visibility into TikTok virality, film/TV sync placements, or radio rotation, which
are the real drivers of outlier numbers.

WHAT TO ADD AT THE END — model export cell:
Add a new code cell at the very end of the notebook:

```python
import joblib, os
os.makedirs("model", exist_ok=True)
joblib.dump(best_rf, "model/rf_pipeline.pkl")
print("Model saved to model/rf_pipeline.pkl")
```

RULES FOR MARKDOWN:
- Never explain what the code does — explain WHY decisions were made
- Never define concepts a DS/MLE already knows (what StandardScaler is, what RMSE means)
- Keep each markdown cell to 3-5 sentences max
- No headers below H2 in markdown cells

After editing, commit:
git add analysis.ipynb
git commit -m "feat: restructure notebook as narrative analysis, add model export"
```

---

### Task 3: Streamlit App (parallel agent)

**Agent prompt:**

```
You are building a Streamlit prediction app at C:\Users\matth\proj\spotify-streams\app.py.
Use the developing-with-streamlit skill.

CONTEXT:
- Portfolio project for a new grad targeting DS/MLE roles
- The app loads a saved sklearn pipeline and predicts Spotify stream counts
- The pipeline is saved at model/rf_pipeline.pkl (joblib format)
- The pipeline is a sklearn Pipeline with a ColumnTransformer preprocessor + RandomForestRegressor
- It expects a pandas DataFrame with exactly these columns:

  Numeric: released_month, released_day, artist_count,
           in_spotify_playlists, in_spotify_charts,
           in_apple_playlists, in_apple_charts,
           in_deezer_playlists, in_deezer_charts,
           in_shazam_charts, danceability_%, valence_%, energy_%,
           acousticness_%, instrumentalness_%, liveness_%, speechiness_%, bpm

  Categorical: key, mode

- Target was log1p(streams), so predictions need np.expm1() to get raw stream count

DATASET MEDIANS (use these as defaults for hidden inputs):
released_month=6, released_day=15, in_apple_charts=5, in_deezer_charts=3,
in_shazam_charts=0, acousticness_%=29, instrumentalness_%=0, liveness_%=10,
speechiness_%=6

APP DESIGN:

Layout: st.set_page_config(layout="wide"), title "Spotify Streams Predictor",
one-line subtitle explaining what it does.

Left sidebar — user inputs:
  st.sidebar.header("Song & Platform Settings")

  Platform exposure (these are the top predictors — label them clearly):
  - in_spotify_playlists: slider 0-10000, default 1000, label "Spotify Playlists"
  - in_apple_playlists: slider 0-5000, default 500, label "Apple Music Playlists"
  - in_deezer_playlists: slider 0-5000, default 300, label "Deezer Playlists"
  - in_spotify_charts: slider 0-200, default 20, label "Spotify Chart Appearances"

  Audio features:
  - danceability_%: slider 0-100, default 65, label "Danceability"
  - energy_%: slider 0-100, default 65, label "Energy"
  - valence_%: slider 0-100, default 50, label "Valence (Positivity)"
  - bpm: slider 60-200, default 120, label "BPM"
  - artist_count: slider 1-4, default 1, label "Number of Artists"

  Song metadata:
  - key: selectbox, options ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B","Unknown"], default "C"
  - mode: selectbox, options ["Major","Minor"], default "Major"

Main panel — prediction output:
  Build a dict from sidebar inputs, fill in median values for hidden features,
  create pd.DataFrame with one row, run pipeline.predict(), apply np.expm1().

  Display:
  1. Large metric: st.metric("Predicted Streams", f"{int(predicted_streams):,}")
  2. Percentile: compute where the prediction falls against these approximate percentile
     thresholds from the training data:
       p25 = 50_000_000
       p50 = 200_000_000
       p75 = 500_000_000
       p90 = 1_000_000_000
     Show: st.caption(f"Approximately top X% of songs in the dataset")
  3. One-line label based on percentile:
       >= p90: "Potential viral hit"
       >= p75: "Major commercial hit"
       >= p50: "Solid chart performer"
       >= p25: "Mid-tier track"
       below p25: "Niche / catalog track"
     Show with st.info()

Load the model once at startup using @st.cache_resource:
  @st.cache_resource
  def load_model():
      return joblib.load("model/rf_pipeline.pkl")

Handle the case where model/rf_pipeline.pkl does not exist yet — show
st.warning("Run analysis.ipynb first to generate model/rf_pipeline.pkl") and st.stop().

Keep the app clean — no extra sections, no about page, no explanatory text beyond
what is needed to use the controls. This is a demo, not a tutorial.

After writing app.py, commit:
git add app.py
git commit -m "feat: add Streamlit prediction app"
```

---

## Task 4: README Update (inline)

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README.md contents**

```markdown
# Spotify Streams Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP.streamlit.app)

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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "feat: overhaul README with live demo link and lean structure"
```

---

## Task 5: User Steps (manual — not automated)

These require a local Jupyter environment with the dataset present.

- [ ] **Step 1: Run analysis.ipynb end to end**

Open Jupyter, run all cells. The final cell saves `model/rf_pipeline.pkl`.

- [ ] **Step 2: Commit the model file**

```bash
git add model/rf_pipeline.pkl
git commit -m "feat: add trained Random Forest pipeline"
```

- [ ] **Step 3: Push to GitHub**

```bash
git push origin main
```

- [ ] **Step 4: Deploy to Streamlit Community Cloud**

1. Go to share.streamlit.io
2. Sign in with GitHub
3. New app → select repo `mzhou709/spotify-streams`, branch `main`, file `app.py`
4. Deploy
5. Copy the assigned URL (e.g. `mzhou709-spotify-streams.streamlit.app`)

- [ ] **Step 5: Update README with real URL**

Replace `https://YOUR-APP.streamlit.app` in README.md with the actual Streamlit URL, then:

```bash
git add README.md
git commit -m "docs: add live Streamlit demo URL"
git push origin main
```

---

## Execution Order

1. Run Task 1 inline (cleanup)
2. Dispatch Task 2 and Task 3 in parallel (notebook + app)
3. Run Task 4 inline (README) once both parallel tasks complete
4. Task 5 is manual — you run the notebook, generate the pkl, deploy
