"""
UFL Rule Impact Dashboard
=========================
Streamlit app — NFL 2025 regular season play-by-play analysis.

Modeling assumption: We are modeling rule impacts, not game theory
responses. Teams are assumed to behave exactly as they did historically.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import nfl_data_py as nfl

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UFL Rule Impact Analyzer",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
BUCKET_LABELS   = ["0–1", "2–3", "4–5", "6–10", "11+"]
BUCKET_BINS     = [0, 1, 3, 5, 10, float("inf")]
SITUATION_ORDER = ["Leading", "Tied", "Trailing (1 score)", "Trailing (2+ scores)"]

REQUIRED_PUNT_COLS = {
    "play_type", "yardline_100", "half_seconds_remaining",
    "posteam", "ydstogo", "score_differential", "epa", "wp", "wpa", "down",
}
REQUIRED_FG_COLS = {"play_type", "kick_distance", "field_goal_result", "posteam"}
REQUIRED_EPA_COLS = {
    "play_type", "down", "ydstogo", "epa", "fourth_down_converted",
    "yardline_100", "half_seconds_remaining",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _check_cols(df: pd.DataFrame, required: set, label: str):
    missing = required - set(df.columns)
    if missing:
        st.error(
            f"❌ Schema change detected in nfl_data_py ({label}). "
            f"Missing columns: `{sorted(missing)}`"
        )
        st.stop()


def score_situation(diff):
    if diff > 0:      return "Leading"
    elif diff == 0:   return "Tied"
    elif diff >= -8:  return "Trailing (1 score)"
    else:             return "Trailing (2+ scores)"


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner="Loading NFL play-by-play data…")
def load_pbp(season: int) -> pd.DataFrame:
    pbp = nfl.import_pbp_data([season], downcast=True)
    return pbp[pbp["season_type"] == "REG"].copy()


# ── Computation: punts ────────────────────────────────────────────────────────
# min_punts is intentionally NOT a cache key here — it only filters the display
# of the team table, so moving it to the UI avoids busting the punt cache every
# time the sidebar slider is nudged.
@st.cache_data(ttl=86400, show_spinner=False)
def compute_punts(season: int, exclude_2min: bool):
    pbp = load_pbp(season)
    _check_cols(pbp, REQUIRED_PUNT_COLS, "punts")

    punts        = pbp[pbp["play_type"] == "punt"].copy()
    punts_in_opp = punts[punts["yardline_100"] < 50]

    if exclude_2min:
        punts_exempt = punts_in_opp[punts_in_opp["half_seconds_remaining"] <= 120]
        banned       = punts_in_opp[punts_in_opp["half_seconds_remaining"] > 120].copy()
    else:
        punts_exempt = pd.DataFrame(columns=punts_in_opp.columns)
        banned       = punts_in_opp.copy()

    n_total  = len(punts)
    n_opp    = len(punts_in_opp)
    n_exempt = len(punts_exempt)
    n_banned = len(banned)

    summary = {
        "total_punts":   n_total,
        "opp_territory": n_opp,
        "exempt":        n_exempt,
        "banned":        n_banned,
        "pct_banned":    n_banned / n_total * 100 if n_total > 0 else 0.0,
    }

    # ── Team table (unfiltered — min_punts applied in UI) ──────────────────────
    banned_by_team = banned.groupby("posteam").size().rename("banned_punts")
    team_table = (
        punts.groupby("posteam").agg(total_punts=("play_type", "count"))
        .join(banned_by_team, how="left")
        .fillna({"banned_punts": 0})
        .assign(
            banned_punts=lambda df: df["banned_punts"].astype(int),
            pct_banned  =lambda df: (df["banned_punts"] / df["total_punts"] * 100).round(1),
        )
        .sort_values("banned_punts", ascending=False)
    )
    team_table.index.name = "Team"
    team_table.columns    = ["Total Punts", "Banned Punts", "% Banned"]

    # ── Situation breakdown ────────────────────────────────────────────────────
    banned_scored = banned.assign(
        situation=banned["score_differential"].apply(score_situation)
    )
    situation_df = (
        banned_scored.groupby("situation")
        .agg(count=("ydstogo", "count"), avg_ydstogo=("ydstogo", "mean"))
        .reindex(SITUATION_ORDER)
        .assign(pct=lambda df: (df["count"] / n_banned * 100).round(1) if n_banned > 0 else 0)
        [["count", "pct", "avg_ydstogo"]]
    )
    situation_df.columns    = ["Count", "% of Banned", "Avg Ydstogo"]
    situation_df["Avg Ydstogo"] = situation_df["Avg Ydstogo"].round(1)
    situation_df.index.name = "Situation"

    return summary, team_table, situation_df, banned


# ── Computation: field goals ───────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def compute_fgs(season: int):
    pbp = load_pbp(season)
    _check_cols(pbp, REQUIRED_FG_COLS, "field goals")

    fgs       = pbp[pbp["play_type"] == "field_goal"].copy()
    long_fgs  = fgs[fgs["kick_distance"] >= 60]
    made_long = long_fgs[long_fgs["field_goal_result"] == "made"]

    n_att  = len(long_fgs)
    n_made = len(made_long)

    summary = {
        "total_attempts": len(fgs),
        "long_attempts":  n_att,
        "long_made":      n_made,
        "make_pct":       n_made / n_att * 100 if n_att > 0 else 0.0,
        "extra_points":   n_made,
    }

    team_fgs = (
        long_fgs.groupby("posteam")
        .agg(
            att  =("field_goal_result", "count"),
            made =("field_goal_result", lambda x: (x == "made").sum()),
        )
        .assign(
            make_pct  =lambda df: (df["made"] / df["att"] * 100).round(1),
            extra_pts =lambda df: df["made"].astype(int),
        )
        .sort_values("extra_pts", ascending=False)
    )
    team_fgs.index.name = "Team"
    team_fgs.columns    = ["Attempts (60+)", "Made (60+)", "Make %", "Extra Pts"]

    return summary, team_fgs


# ── Computation: EPA swing ─────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def compute_epa_swing(season: int, exclude_2min: bool):
    pbp = load_pbp(season)
    _check_cols(pbp, REQUIRED_EPA_COLS, "EPA swing")

    punts_in_opp = pbp[(pbp["play_type"] == "punt") & (pbp["yardline_100"] < 50)]
    banned = (
        punts_in_opp[punts_in_opp["half_seconds_remaining"] > 120].copy()
        if exclude_2min else punts_in_opp.copy()
    )

    fourth_go = pbp[
        (pbp["down"] == 4) & (pbp["play_type"].isin(["run", "pass"]))
    ].copy()

    for df in (fourth_go, banned):
        df["bucket"] = pd.cut(
            df["ydstogo"], bins=BUCKET_BINS,
            labels=BUCKET_LABELS, right=True, include_lowest=True,
        )

    _attempts = (
        fourth_go.groupby("bucket", observed=True)["fourth_down_converted"]
        .agg(go_attempts="count", conv_rate="mean")
    )
    _epa_conv = (
        fourth_go[fourth_go["fourth_down_converted"] == 1]
        .groupby("bucket", observed=True)["epa"].mean()
        .rename("epa_if_converted")
    )
    _epa_fail = (
        fourth_go[fourth_go["fourth_down_converted"] == 0]
        .groupby("bucket", observed=True)["epa"].mean()
        .rename("epa_if_failed")
    )
    _punt_epa = (
        banned.groupby("bucket", observed=True)["epa"].mean()
        .rename("avg_punt_epa")
    )

    tbl = _attempts.join(_epa_conv).join(_epa_fail).join(_punt_epa).reindex(BUCKET_LABELS)
    tbl["exp_epa_go"] = (
        tbl["conv_rate"] * tbl["epa_if_converted"]
        + (1 - tbl["conv_rate"]) * tbl["epa_if_failed"]
    )
    tbl["epa_swing"] = tbl["exp_epa_go"] - tbl["avg_punt_epa"]

    tbl.index.name = "Yards to Go"
    tbl.columns    = [
        "Go Attempts", "Conv Rate", "EPA if Conv",
        "EPA if Failed", "Exp EPA (Go)", "Avg Punt EPA", "EPA Swing",
    ]
    return tbl


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    season = st.selectbox("Season", [2025, 2024, 2023], index=0)
    exclude_2min = st.checkbox(
        "Exclude punts inside 2-minute warning", value=True,
        help="When checked, punts that occurred inside the final 2 minutes of either half "
             "are excluded from the 'banned' count (they are allowed by the UFL rule exemption).",
    )
    min_punts = st.number_input(
        "Min punts to show team (punt table)", min_value=0, value=0, step=5,
    )

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("UFL Rule Impact — NFL Regular Season Analysis")
st.caption(
    "**Modeling assumption:** We are modeling rule impacts, not game theory responses. "
    "Teams are assumed to behave exactly as they did historically."
)
st.divider()

# ── Load & compute (errors surface here, outside cache) ───────────────────────
try:
    punt_summary, team_punt_table, situation_df, banned_df = compute_punts(
        season, exclude_2min
    )
    fg_summary, team_fg_table = compute_fgs(season)
    epa_table = compute_epa_swing(season, exclude_2min)
except Exception as e:
    st.error(f"❌ Data error: {e}")
    st.stop()

# Apply min_punts filter here (in UI) so it doesn't bust the compute cache
team_punt_display = (
    team_punt_table[team_punt_table["Total Punts"] >= min_punts]
    if min_punts > 0 else team_punt_table
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Punts", "Field Goals", "4th Down Decision"])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — PUNTS
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Punt Rule Impact")
    st.caption(
        "UFL rule: no punts inside the opponent's 50-yard line. "
        "Exemption applies inside the 2-minute warning of each half."
    )

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Punts",        f"{punt_summary['total_punts']:,}")
    c2.metric("In Opp Territory",   f"{punt_summary['opp_territory']:,}")
    c3.metric("Banned Punts",       f"{punt_summary['banned']:,}")
    c4.metric("% Would Be Banned",  f"{punt_summary['pct_banned']:.1f}%")

    if exclude_2min and punt_summary["exempt"]:
        st.caption(
            f"*{punt_summary['exempt']:,} opp-territory punts excluded — "
            f"occurred inside the 2-minute warning and are allowed under the exemption.*"
        )

    st.divider()

    # Charts
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Banned Punts by Score Situation")
        bar_data = situation_df.dropna(subset=["Count"]).reset_index()
        fig_bar = px.bar(
            bar_data,
            x="Situation", y="Count",
            text="Count",
            color="Situation",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            showlegend=False,
            xaxis_title="Score Situation at Time of Punt",
            yaxis_title="Banned Punts",
            margin=dict(t=10, b=0),
            yaxis=dict(range=[0, bar_data["Count"].max() * 1.25]),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.markdown("#### Yards to Go Distribution")
        ydstogo_vals = banned_df["ydstogo"].dropna()
        mean_val   = ydstogo_vals.mean()
        median_val = ydstogo_vals.median()

        fig_hist = px.histogram(
            x=ydstogo_vals, nbins=25,
            labels={"x": "Yards to Go", "y": "Frequency"},
            color_discrete_sequence=["#636EFA"],
        )
        fig_hist.add_vline(
            x=mean_val, line_dash="dash", line_color="firebrick",
            annotation_text=f"Mean: {mean_val:.1f}",
            annotation_position="top right",
        )
        fig_hist.add_vline(
            x=median_val, line_dash="dot", line_color="navy",
            annotation_text=f"Median: {median_val:.1f}",
            annotation_position="top left",
        )
        fig_hist.update_layout(
            showlegend=False, margin=dict(t=10, b=0), yaxis_title="Frequency",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()
    st.markdown("#### Situation Breakdown")
    st.dataframe(situation_df, use_container_width=True)

    st.divider()
    st.markdown("#### Banned Punts by Team")
    st.dataframe(team_punt_display, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — FIELD GOALS
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Field Goal Rule Impact")
    st.caption(
        "UFL rule: field goals from 60+ yards are worth **4 points** instead of 3. "
        "Each made 60+ yard FG generates +1 extra point under the new rule."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total FG Attempts",    f"{fg_summary['total_attempts']:,}")
    c2.metric("60+ Yard Attempts",    f"{fg_summary['long_attempts']:,}")
    c3.metric("60+ Yard Makes",       f"{fg_summary['long_made']:,}")
    c4.metric("Extra Points (League)", f"+{fg_summary['extra_points']:,}")
    st.caption(f"*Make % from 60+ yards: {fg_summary['make_pct']:.1f}%*")

    st.divider()
    st.markdown("#### 60+ Yard FGs by Team")
    st.dataframe(team_fg_table, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — 4TH DOWN DECISION / EPA SWING
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("EPA Swing: Go For It vs Punt")
    st.caption(
        "For each yards-to-go bucket, compares the **expected EPA of going for it** "
        "(weighted by historical 4th-down conversion rates) against the "
        "**actual EPA of the banned punt plays**."
    )

    st.info(
        "**Assumptions:** Conversion rates are from this season's 4th-down run/pass plays. "
        "Expected EPA = P(convert) × EPA_success + P(fail) × EPA_failure. "
        "No adaptive strategy is assumed — teams play exactly as they did historically."
    )

    st.divider()

    # Format display copy (keep raw floats for styling, format after)
    epa_display = epa_table.copy()

    try:
        import matplotlib  # noqa: F401 — required by pandas Styler background_gradient
        styled = (
            epa_display.style
            .format("{:.1%}", subset=["Conv Rate"])
            .format("{:.0f}", subset=["Go Attempts"])
            .format("{:+.3f}", subset=["EPA if Conv", "EPA if Failed", "Exp EPA (Go)", "Avg Punt EPA", "EPA Swing"])
            .background_gradient(subset=["EPA Swing"], cmap="RdYlGn", vmin=-1.5, vmax=1.5)
        )
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(epa_display, use_container_width=True)
        st.caption("Styling unavailable in this environment; showing plain table.")

    st.markdown("""
| EPA Swing | Interpretation |
|-----------|----------------|
| **Positive (green)** | Going for it expected to produce more EPA than punting in that distance range |
| **Near zero** | Decision is roughly equivalent in EPA terms |
| **Negative (red)** | Punting was actually the higher-EPA decision in that distance range |
""")
