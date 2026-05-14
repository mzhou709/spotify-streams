import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(layout="wide")

st.title("Spotify Streams Predictor")
st.write("Adjust the song and platform settings to predict how many streams a track might accumulate.")


@st.cache_resource
def load_model():
    return joblib.load("model/rf_pipeline.pkl")


try:
    model = load_model()
except Exception:
    st.warning("Run analysis.ipynb first to generate model/rf_pipeline.pkl")
    st.stop()

# --- Sidebar inputs ---
st.sidebar.header("Song & Platform Settings")

st.sidebar.subheader("Platform Exposure")
in_spotify_playlists = st.sidebar.slider("Spotify Playlists", 0, 10000, 1000)
in_apple_playlists = st.sidebar.slider("Apple Music Playlists", 0, 5000, 500)
in_deezer_playlists = st.sidebar.slider("Deezer Playlists", 0, 5000, 300)
in_spotify_charts = st.sidebar.slider("Spotify Chart Appearances", 0, 200, 20)

st.sidebar.subheader("Audio Features")
danceability = st.sidebar.slider("Danceability", 0, 100, 65)
energy = st.sidebar.slider("Energy", 0, 100, 65)
valence = st.sidebar.slider("Valence (Positivity)", 0, 100, 50)
bpm = st.sidebar.slider("BPM", 60, 200, 120)
artist_count = st.sidebar.slider("Number of Artists", 1, 4, 1)

st.sidebar.subheader("Song Metadata")
key = st.sidebar.selectbox(
    "Key",
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Unknown"],
    index=0,
)
mode = st.sidebar.selectbox("Mode", ["Major", "Minor"], index=0)

# --- Build input DataFrame ---
input_data = {
    # User-provided
    "in_spotify_playlists": in_spotify_playlists,
    "in_apple_playlists": in_apple_playlists,
    "in_deezer_playlists": in_deezer_playlists,
    "in_spotify_charts": in_spotify_charts,
    "danceability_%": danceability,
    "energy_%": energy,
    "valence_%": valence,
    "bpm": bpm,
    "artist_count": artist_count,
    "key": key,
    "mode": mode,
    # Hidden — dataset medians
    "released_month": 6,
    "released_day": 15,
    "in_apple_charts": 5,
    "in_deezer_charts": 3,
    "in_shazam_charts": 0,
    "acousticness_%": 29,
    "instrumentalness_%": 0,
    "liveness_%": 10,
    "speechiness_%": 6,
}

column_order = [
    "released_month", "released_day", "artist_count",
    "in_spotify_playlists", "in_spotify_charts",
    "in_apple_playlists", "in_apple_charts",
    "in_deezer_playlists", "in_deezer_charts",
    "in_shazam_charts",
    "danceability_%", "valence_%", "energy_%",
    "acousticness_%", "instrumentalness_%", "liveness_%", "speechiness_%",
    "bpm",
    "key", "mode",
]

df_input = pd.DataFrame([input_data])[column_order]

# --- Predict ---
log_pred = model.predict(df_input)[0]
predicted_streams = np.expm1(log_pred)

PERCENTILE_TIERS = [
    (1_000_000_000, "top 10%",    "Potential viral hit"),
    (500_000_000,   "top 25%",    "Major commercial hit"),
    (200_000_000,   "top 50%",    "Solid chart performer"),
    (50_000_000,    "top 75%",    "Mid-tier track"),
    (0,             "bottom 25%", "Niche / catalog track"),
]

for threshold, pct, label in PERCENTILE_TIERS:
    if predicted_streams >= threshold:
        percentile_text = f"Approximately {pct} of songs in the dataset"
        break

# --- Display ---
st.metric("Predicted Streams", f"{int(predicted_streams):,}")
st.caption(percentile_text)
st.info(label)
