import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="NBA MVP Predictor — Dashboard", page_icon="🏀", layout="wide")

st.title("🏀 NBA MVP Predictor — Interactive Dashboard")
st.write(
    """
This app visualizes predictions from your repo's **output.csv** and lets you:
- Explore **Top-10 MVP predictions** by season
- See **Actual vs Predicted** with metrics (R², MAE) if actuals exist
- Drill down on a **single player** across seasons

> **Tip:** Place an `output.csv` in the repo root (or upload below). Columns are auto-detected.
    """
)

# -----------------------------
# Helpers
# -----------------------------
def load_data():
    """Load from local output.csv if present; allow overriding via uploader."""
    default_path = "output.csv"
    df_local = None
    if os.path.exists(default_path):
        try:
            df_local = pd.read_csv(default_path)
        except Exception as e:
            st.warning(f"Could not read local output.csv: {e}")

    uploaded = st.file_uploader("Upload output.csv (optional, overrides local)", type=["csv"])
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Uploaded CSV could not be read: {e}")
            st.stop()

    return df_local

CANDIDATE_COLS = {
    "player": ["Player", "Plyr", "player_name", "Name", "PLAYER"],
    "season": ["Season", "Szn", "season", "Year", "year"],
    "pred":   ["Predicted", "pred", "y_pred", "prediction", "pred_award_share", "Pred_Score"],
    "actual": ["MVP_Shr", "Actual", "actual", "y_true", "award_share", "Award Share"],
}

def find_col(df, keys):
    # try exact, then case-insensitive
    for k in keys:
        if k in df.columns:
            return k
    lower = {c.lower(): c for c in df.columns}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    return None

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    m, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return pd.Series(0, index=s.index)
    return (s - m) / sd

# -----------------------------
# Data
# -----------------------------
df = load_data()
if df is None:
    st.info("Awaiting data… Add an `output.csv` to the repo or upload a CSV.")
    st.stop()

# Remove exact duplicate header names (keep first occurrence)
df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

# Detect columns (supports Plyr / Szn / MVP_Shr)
player_col = find_col(df, CANDIDATE_COLS["player"]) or st.selectbox("Select player column", options=df.columns)
season_col = find_col(df, CANDIDATE_COLS["season"]) or st.selectbox("Select season column", options=df.columns)
pred_col   = find_col(df, CANDIDATE_COLS["pred"])
actual_col = find_col(df, CANDIDATE_COLS["actual"])  # optional

work = df.copy()

# If there’s no proper prediction column, synthesize one so the app still works
if pred_col is None or pred_col == player_col:
    candidates = [c for c in ["PER", "WS", "BPM"] if c in work.columns]
    if candidates:
        work["Pred_Score"] = sum(zscore(work[c]) for c in candidates)
        pred_col = "Pred_Score"
    else:
        pts_col = next((c for c in ["PTS", "points"] if c in work.columns), None)
        if pts_col is not None:
            work["Pred_Score"] = zscore(work[pts_col])
            pred_col = "Pred_Score"
        else:
            st.error("No prediction column found and couldn't synthesize one (missing PER/WS/BPM/PTS).")
            st.stop()

# Coerce numerics where needed
for c in [pred_col, actual_col]:
    if c and c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")

# season -> string for grouping/selection
work[season_col] = work[season_col].astype(str)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Filters")
    seasons = sorted(work[season_col].dropna().unique())
    sel_season = st.selectbox("Season", options=seasons, index=max(0, len(seasons)-1))
    top_n = st.slider("Top N leaderboard", 5, 20, 10)
    st.caption("Change the predicted column in code or add your own to the CSV if needed.")

# -----------------------------
# Leaderboard by season
# -----------------------------
season_df = work[work[season_col] == sel_season].dropna(subset=[pred_col])
leaderboard_cols = [player_col, pred_col] + ([actual_col] if (actual_col and actual_col in season_df.columns) else [])
leaderboard = (
    season_df[leaderboard_cols]
    .sort_values(by=pred_col, ascending=False)
    .head(top_n)
)

c1, c2 = st.columns([1.2, 1])
with c1:
    st.subheader(f"Top {top_n} — {sel_season}")
    if leaderboard.empty:
        st.warning("No rows for this season.")
    else:
        fig = px.bar(
            leaderboard,
            x=pred_col,
            y=player_col,
            orientation="h",
            title=f"Predicted MVP Score — {sel_season}",
            text=pred_col,
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Download season table")
    st.download_button(
        label="Download CSV",
        data=leaderboard.to_csv(index=False).encode("utf-8"),
        file_name=f"mvp_leaderboard_{sel_season}.csv",
        mime="text/csv",
    )
    st.dataframe(leaderboard.reset_index(drop=True))

st.markdown("---")

# -----------------------------
# Parity: Actual vs Predicted
# -----------------------------
if actual_col and actual_col in work.columns:
    clean = work.dropna(subset=[pred_col, actual_col])
    if not clean.empty:
        r2 = r2_score(clean[actual_col], clean[pred_col])
        mae = mean_absolute_error(clean[actual_col], clean[pred_col])
        st.subheader("Actual vs Predicted (all seasons)")
        # trendline='ols' requires statsmodels; fall back if missing
        try:
            fig2 = px.scatter(
                clean,
                x=actual_col,
                y=pred_col,
                hover_data={player_col: True, season_col: True},
                trendline="ols",
                title=f"Parity Plot — R²={r2:.3f}, MAE={mae:.3f}"
            )
        except Exception:
            fig2 = px.scatter(
                clean,
                x=actual_col,
                y=pred_col,
                hover_data={player_col: True, season_col: True},
                title=f"Parity Plot — R²={r2:.3f}, MAE={mae:.3f} (no trendline)"
            )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Need both actual and predicted columns to draw the parity plot.")
else:
    st.info("No actual column detected — parity plot and metrics hidden.")

st.markdown("---")

# -----------------------------
# Player drill-down
# -----------------------------
players = sorted(work[player_col].dropna().unique())
sel_player = st.selectbox("Player drill-down", options=players)
pdf = work[work[player_col] == sel_player].sort_values(by=season_col)

st.subheader(f"{sel_player} — Predicted vs Actual over seasons")
if actual_col and actual_col in pdf.columns:
    long_df = pd.melt(
        pdf[[season_col, pred_col, actual_col]],
        id_vars=[season_col],
        value_vars=[pred_col, actual_col],
        var_name="Type",
        value_name="Score",
    )
else:
    long_df = pd.melt(
        pdf[[season_col, pred_col]],
        id_vars=[season_col],
        value_vars=[pred_col],
        var_name="Type",
        value_name="Score",
    )

fig3 = px.line(long_df, x=season_col, y="Score", color="Type", markers=True)
st.plotly_chart(fig3, use_container_width=True)

st.caption("Built with Streamlit • Plotly • scikit-learn. Columns are auto-detected; upload a CSV to override.")
