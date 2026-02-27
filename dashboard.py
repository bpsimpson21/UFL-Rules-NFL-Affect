"""
UFL Rule Impact Dashboard
=========================
Streamlit app — NFL 2025 regular season play-by-play analysis.

Modeling assumption: We are modeling rule impacts, not game theory
responses. Teams are assumed to behave exactly as they did historically.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import nfl_data_py as nfl

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UFL Rule Impact Analyzer",
    page_icon="🎙️",
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
REQUIRED_SNEAK_COLS = {
    "rush_attempt", "pass_attempt", "rusher_player_id", "passer_player_id",
    "posteam", "down", "ydstogo", "score_differential",
    "third_down_converted", "fourth_down_converted", "game_id",
    "home_team", "total_home_score", "total_away_score",
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


# ── Computation: tush push ─────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def compute_tush_push(season: int):
    pbp = load_pbp(season)
    _check_cols(pbp, REQUIRED_SNEAK_COLS, "tush push")

    # ── Identify QB sneaks ────────────────────────────────────────────────────
    # Prefer qb_sneak column if it exists and is actually populated
    if "qb_sneak" in pbp.columns and int(pbp["qb_sneak"].sum()) > 0:
        sneaks = pbp[pbp["qb_sneak"] == 1].copy()
    else:
        # Heuristic: short-yardage rush where the rusher is the team's primary QB.
        # Primary QB = player with the most pass attempts for that team in that game.
        qb_by_game = (
            pbp[pbp["pass_attempt"] == 1]
            .groupby(["game_id", "posteam"])["passer_player_id"]
            .agg(lambda x: x.dropna().mode().iloc[0] if x.dropna().size > 0 else None)
            .reset_index()
            .rename(columns={"passer_player_id": "primary_qb_id"})
        )
        rushes = (
            pbp[(pbp["rush_attempt"] == 1) & (pbp["ydstogo"] == 1)]
            .merge(qb_by_game, on=["game_id", "posteam"], how="left")
        )
        sneaks = rushes[
            rushes["rusher_player_id"] == rushes["primary_qb_id"]
        ].copy()

    # 3rd/4th down conversions (where "conversion" is meaningful)
    sneaks_crit = sneaks[sneaks["down"].isin([3, 4])].copy()
    sneaks_crit["converted"] = (
        ((sneaks_crit["down"] == 3) & (sneaks_crit["third_down_converted"] == 1)) |
        ((sneaks_crit["down"] == 4) & (sneaks_crit["fourth_down_converted"] == 1))
    )
    league_conv = sneaks_crit["converted"].mean() if len(sneaks_crit) > 0 else 0.0

    # Team counts (all downs), sorted ascending so highest lands at top of h-bar
    team_counts = (
        sneaks.groupby("posteam").size()
        .sort_values(ascending=True)
        .reset_index()
    )
    team_counts.columns = ["Team", "QB Sneaks"]
    team_counts["color"] = [
        "#004c54" if t == "PHI" else "#b0b8c1" for t in team_counts["Team"]
    ]

    # Eagles subset
    eagles = sneaks[sneaks["posteam"] == "PHI"].copy()
    eagles_crit = sneaks_crit[sneaks_crit["posteam"] == "PHI"].copy()
    eagles_conv = eagles_crit["converted"].mean() if len(eagles_crit) > 0 else 0.0

    # Eagles by down
    _down_labels = {1: "1st Down", 2: "2nd Down", 3: "3rd Down", 4: "4th Down"}
    eagles_by_down = (
        eagles.groupby("down").size()
        .rename("Count")
        .reset_index()
    )
    eagles_by_down["Down"] = eagles_by_down["down"].map(_down_labels)
    eagles_by_down = eagles_by_down[["Down", "Count"]]

    # Eagles by ydstogo bucket
    eagles["bucket"] = pd.cut(
        eagles["ydstogo"], bins=BUCKET_BINS,
        labels=BUCKET_LABELS, right=True, include_lowest=True,
    )
    eagles_by_ydstogo = (
        eagles.groupby("bucket", observed=True).size()
        .rename("Count")
        .reindex(BUCKET_LABELS)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    eagles_by_ydstogo.columns = ["Yards to Go", "Count"]

    # Eagles by game situation
    eagles["situation"] = eagles["score_differential"].apply(score_situation)
    eagles_by_sit = (
        eagles.groupby("situation").size()
        .rename("Count")
        .reindex(SITUATION_ORDER)
        .reset_index()
    )
    eagles_by_sit.columns = ["Situation", "Count"]

    # Games won with a critical converted sneak (3rd/4th down, Eagles, converted)
    critical_game_ids = set(eagles_crit[eagles_crit["converted"]]["game_id"].unique())
    final = pbp.groupby("game_id").agg(
        home_team=("home_team", "first"),
        final_home=("total_home_score", "max"),
        final_away=("total_away_score", "max"),
    )
    wins_crit = sum(
        (
            (final.loc[gid, "home_team"] == "PHI" and
             final.loc[gid, "final_home"] > final.loc[gid, "final_away"])
            or
            (final.loc[gid, "home_team"] != "PHI" and
             final.loc[gid, "final_away"] > final.loc[gid, "final_home"])
        )
        for gid in critical_game_ids if gid in final.index
    )

    return {
        "n_total":       len(sneaks),
        "league_conv":   league_conv,
        "n_eagles":      len(eagles),
        "eagles_conv":   eagles_conv,
        "n_crit_games":  len(critical_game_ids),
        "wins_crit":     wins_crit,
        "team_counts":   team_counts,
        "by_down":       eagles_by_down,
        "by_ydstogo":    eagles_by_ydstogo,
        "by_situation":  eagles_by_sit,
    }



# ══════════════════════════════════════════════════════════════════════════════
# BRAND CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
TEAL     = "#0fbcce"
GOLD     = "#c8a84b"
CC_GREEN = "#006a4e"
BG       = "#ffffff"
CARD     = "#f4f6f8"
SIDEBAR_BG = "#f0f2f5"
TEXT     = "#1e2329"
MUTED    = "#5c6370"

CHART_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD,
    font=dict(color=TEXT, family="sans-serif"),
    xaxis=dict(gridcolor="rgba(0,0,0,0.07)", linecolor="rgba(0,0,0,0.13)", tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor="rgba(0,0,0,0.07)", linecolor="rgba(0,0,0,0.13)", tickfont=dict(color=MUTED)),
    margin=dict(t=10, b=0, l=0, r=0),
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
/* ── Global background & text ───────────────────────────────────────── */
[data-testid="stAppViewContainer"] {{
    background-color: {BG};
}}
[data-testid="stHeader"] {{
    background-color: {BG};
    border-bottom: 2px solid {CC_GREEN};
}}
body, .stMarkdown, .stText, p, li {{
    color: {TEXT};
}}

/* ── Page border ────────────────────────────────────────────────────── */
.block-container {{
    border: 3px solid {CC_GREEN};
    border-radius: 10px;
    padding: 2rem 3rem !important;
    margin-top: 1rem;
}}

/* ── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG};
    border-right: 2px solid {CC_GREEN};
}}
[data-testid="stSidebar"] * {{
    color: {TEXT} !important;
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stNumberInput label {{
    color: {MUTED} !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}}

/* ── Tabs ───────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background-color: transparent;
    border-bottom: 1px solid {CC_GREEN}55;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    color: {MUTED};
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-radius: 4px 4px 0 0;
    padding: 8px 22px;
    border: none;
}}
.stTabs [aria-selected="true"] {{
    color: {TEAL} !important;
    background-color: {TEAL}14 !important;
    border-bottom: 2px solid {TEAL} !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT} !important;
    background-color: {TEAL}0d !important;
}}

/* ── KPI metric cards ───────────────────────────────────────────────── */
[data-testid="metric-container"] {{
    background-color: {CARD};
    border: 1px solid {TEAL}44;
    border-radius: 10px;
    padding: 18px 22px 14px 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
[data-testid="metric-container"] label,
[data-testid="stMetricLabel"] {{
    color: {MUTED} !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
}}
[data-testid="stMetricValue"] > div {{
    color: {TEAL} !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
}}

/* ── Section headings (####) ────────────────────────────────────────── */
h4 {{
    color: {GOLD} !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
    margin-top: 1.2rem !important;
}}

/* ── Dividers ───────────────────────────────────────────────────────── */
hr {{
    border-color: {CC_GREEN}44 !important;
}}

/* ── Dataframe container ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {CC_GREEN}44;
    border-radius: 8px;
    overflow: hidden;
}}

/* ── Info / caption boxes ───────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {{
    color: {MUTED} !important;
    font-size: 0.78rem !important;
}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<p style='color:{TEAL};font-weight:800;font-size:0.8rem;"
        f"letter-spacing:0.1em;text-transform:uppercase;margin-bottom:16px;'>"
        f"🎙️ GUS &amp; DAVE SPORTS</p>",
        unsafe_allow_html=True,
    )
    season = st.selectbox("Season", [2025, 2024, 2023], index=0)
    exclude_2min = st.checkbox(
        "Exclude punts inside 2-minute warning", value=True,
        help="When checked, punts that occurred inside the final 2 minutes of either half "
             "are excluded from the 'banned' count (they are allowed by the UFL rule exemption).",
    )
    min_punts = st.number_input(
        "Min punts to show team (punt table)", min_value=0, value=0, step=5,
    )
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.68rem;margin-top:32px;'>Created by Brennan Simpson</p>",
        unsafe_allow_html=True,
    )

# ── Brand header ───────────────────────────────────────────────────────────────
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "gd_logo.png")
header_left, header_right = st.columns([3, 1])

with header_left:
    logo_shown = False
    if os.path.exists(LOGO_PATH):
        logo_col, title_col = st.columns([1, 6])
        with logo_col:
            st.image(LOGO_PATH, width=64)
        with title_col:
            st.markdown(
                f"<h1 style='color:{TEXT};font-weight:900;font-size:1.75rem;"
                f"letter-spacing:-0.01em;margin:0;padding:0;line-height:1.15;'>"
                f"UFL RULE IMPACT</h1>"
                f"<p style='color:{MUTED};font-size:0.85rem;margin:2px 0 0 0;'>"
                f"NFL Regular Season Analysis</p>",
                unsafe_allow_html=True,
            )
        logo_shown = True
    if not logo_shown:
        st.markdown(
            f"<h1 style='color:{TEXT};font-weight:900;font-size:1.75rem;"
            f"letter-spacing:-0.01em;margin:0;padding:0;line-height:1.15;'>"
            f"UFL RULE IMPACT</h1>"
            f"<p style='color:{MUTED};font-size:0.85rem;margin:2px 0 0 0;'>"
            f"NFL Regular Season Analysis</p>",
            unsafe_allow_html=True,
        )

with header_right:
    st.markdown(
        f"<div style='text-align:right;padding-top:6px;'>"
        f"<p style='color:{TEAL};font-weight:800;font-size:0.72rem;"
        f"letter-spacing:0.1em;text-transform:uppercase;margin:0;'>"
        f"🎙️ Gus &amp; Dave Sports Podcast</p>"
        f"<p style='color:{MUTED};font-size:0.68rem;margin:3px 0 0 0;'>"
        f"Created by Brennan Simpson</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    f"<p style='color:{MUTED};font-size:0.78rem;margin:10px 0 4px 0;'>"
    f"<em>Modeling assumption: We are modeling rule impacts, not game theory responses. "
    f"Teams are assumed to behave exactly as they did historically.</em></p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Load & compute (errors surface here, outside cache) ───────────────────────
try:
    punt_summary, team_punt_table, situation_df, banned_df = compute_punts(
        season, exclude_2min
    )
    fg_summary, team_fg_table = compute_fgs(season)
    epa_table = compute_epa_swing(season, exclude_2min)
    tush_push_data = compute_tush_push(season)
except Exception as e:
    st.error(f"❌ Data error: {e}")
    st.stop()

# Apply min_punts filter here (in UI) so it doesn't bust the compute cache
team_punt_display = (
    team_punt_table[team_punt_table["Total Punts"] >= min_punts]
    if min_punts > 0 else team_punt_table
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🦵  Punts", "🎯  Field Goals", "📊  4th Down Decision", "🏈  Tush Push"])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — PUNTS
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.8rem;margin-bottom:14px;'>"
        f"UFL rule: no punts inside the opponent's 50-yard line. "
        f"Exemption applies inside the 2-minute warning of each half.</p>",
        unsafe_allow_html=True,
    )

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Punts",       f"{punt_summary['total_punts']:,}")
    c2.metric("In Opp Territory",  f"{punt_summary['opp_territory']:,}")
    c3.metric("Banned Punts",      f"{punt_summary['banned']:,}")
    c4.metric("% Would Be Banned", f"{punt_summary['pct_banned']:.1f}%")

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
            x="Situation",
            y="Count",
            text="Count",
            color="Situation",
            color_discrete_sequence=[TEAL, GOLD, "#4a9ebb"],
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            showlegend=False,
            xaxis_title="Score Situation at Time of Punt",
            **CHART_LAYOUT,
        )
        fig_bar.update_yaxes(title="Banned Punts", range=[0, bar_data["Count"].max() * 1.3])
        st.plotly_chart(fig_bar, width="stretch")

    with col_r:
        st.markdown("#### Yards to Go Distribution")
        ydstogo_vals = banned_df["ydstogo"].dropna()
        mean_val   = ydstogo_vals.mean()
        median_val = ydstogo_vals.median()

        fig_hist = px.histogram(
            x=ydstogo_vals, nbins=25,
            labels={"x": "Yards to Go", "y": "Frequency"},
            color_discrete_sequence=[TEAL],
        )
        fig_hist.add_vline(
            x=mean_val, line_dash="dash", line_color=GOLD, line_width=2,
            annotation_text=f"Mean: {mean_val:.1f}",
            annotation_font=dict(color=GOLD, size=11),
            annotation_position="top right",
        )
        fig_hist.add_vline(
            x=median_val, line_dash="dot", line_color="#a0cfdf", line_width=2,
            annotation_text=f"Median: {median_val:.1f}",
            annotation_font=dict(color="#a0cfdf", size=11),
            annotation_position="top left",
        )
        fig_hist.update_layout(
            showlegend=False,
            yaxis_title="Frequency",
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig_hist, width="stretch")

    st.divider()
    st.markdown("#### Situation Breakdown")
    st.dataframe(situation_df, width="stretch")

    st.divider()
    st.markdown("#### Banned Punts by Team")
    st.dataframe(team_punt_display, width="stretch")


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — FIELD GOALS
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.8rem;margin-bottom:14px;'>"
        f"UFL rule: field goals from 60+ yards are worth <strong>4 points</strong> instead of 3. "
        f"Each made 60+ yard FG generates +1 extra point under the new rule.</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total FG Attempts",     f"{fg_summary['total_attempts']:,}")
    c2.metric("60+ Yard Attempts",     f"{fg_summary['long_attempts']:,}")
    c3.metric("60+ Yard Makes",        f"{fg_summary['long_made']:,}")
    c4.metric("Extra Points (League)", f"+{fg_summary['extra_points']:,}")
    st.caption(f"*Make % from 60+ yards: {fg_summary['make_pct']:.1f}%*")

    st.divider()
    st.markdown("#### 60+ Yard FGs by Team")
    st.dataframe(team_fg_table, width="stretch")


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — 4TH DOWN DECISION / EPA SWING
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.8rem;margin-bottom:6px;'>"
        f"For each yards-to-go bucket, compares the <strong>expected EPA of going for it</strong> "
        f"(weighted by historical 4th-down conversion rates) against the "
        f"<strong>actual EPA of the banned punt plays</strong>.</p>",
        unsafe_allow_html=True,
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
        st.dataframe(styled, width="stretch")
    except Exception:
        st.dataframe(epa_display, width="stretch")
        st.caption("Styling unavailable in this environment; showing plain table.")

    st.markdown(f"""
| EPA Swing | Interpretation |
|-----------|----------------|
| **Positive (green)** | Going for it expected to produce more EPA than punting |
| **Near zero** | Decision is roughly equivalent in EPA terms |
| **Negative (red)** | Punting was the higher-EPA decision in that distance range |
""")

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — TUSH PUSH
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.8rem;margin-bottom:14px;'>"
        f"The UFL has banned the &ldquo;tush push&rdquo; (QB sneak with pushing assistance). "
        f"We can&rsquo;t identify confirmed tush pushes without reviewing every play, "
        f"so this tab analyzes <strong>all QB rushing attempts from exactly 1 yard out</strong> "
        f"&mdash; the distance where tush pushes almost exclusively occur. "
        f"No team would lose more under this rule than the "
        f"<strong>Philadelphia Eagles</strong>.</p>",
        unsafe_allow_html=True,
    )

    tp = tush_push_data

    # ── League-wide ────────────────────────────────────────────────────────
    st.markdown("#### League-Wide QB Sneaks")
    c1, c2 = st.columns(2)
    c1.metric("Total QB Rushes (1 Yd Out)", f"{tp['n_total']:,}")
    c2.metric("Conv Rate on 3rd / 4th Down", f"{tp['league_conv']:.1%}")

    st.markdown("#### QB Rushes from 1 Yard Out — by Team")
    fig_teams = px.bar(
        tp["team_counts"],
        x="QB Sneaks",
        y="Team",
        orientation="h",
        color="color",
        color_discrete_map="identity",
        text="QB Sneaks",
    )
    fig_teams.update_traces(textposition="outside", textfont=dict(color=TEXT, size=11))
    fig_teams.update_layout(
        showlegend=False,
        xaxis_title="Number of QB Sneaks",
        yaxis_title="",
        height=max(400, len(tp["team_counts"]) * 22),
        **CHART_LAYOUT,
    )
    fig_teams.update_xaxes(range=[0, tp["team_counts"]["QB Sneaks"].max() * 1.25])
    st.plotly_chart(fig_teams, width="stretch")

    # ── Eagles deep-dive ───────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Eagles Deep-Dive 🦅")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eagles QB Rushes (1 Yd Out)", f"{tp['n_eagles']:,}")
    c2.metric("Eagles Conv Rate (3rd/4th)",  f"{tp['eagles_conv']:.1%}")
    c3.metric("League Conv Rate (3rd/4th)",  f"{tp['league_conv']:.1%}")
    c4.metric("Wins w/ 1-Yd Conversion",     f"{tp['wins_crit']}")

    st.caption(
        f"*Eagles had a converted QB rush from 1 yard out on 3rd or 4th down in "
        f"{tp['n_crit_games']} game(s) and won {tp['wins_crit']} of them. "
        f"These plays are not confirmed tush pushes — they are all 1-yard QB rushes.*"
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Eagles QB Rushes (1 Yd Out) by Down")
        _max_down = max(tp["by_down"]["Count"].max(), 1)
        fig_down = px.bar(
            tp["by_down"],
            x="Down", y="Count",
            text="Count",
            color_discrete_sequence=["#004c54"],
        )
        fig_down.update_traces(textposition="outside")
        fig_down.update_layout(showlegend=False, xaxis_title="", **CHART_LAYOUT)
        fig_down.update_yaxes(title="Count", range=[0, _max_down * 1.3])
        st.plotly_chart(fig_down, width="stretch")

    with col_r:
        st.markdown("#### Eagles QB Rushes (1 Yd Out) by Game Situation")
        _sit_data = tp["by_situation"].dropna(subset=["Count"])
        _max_sit = max(_sit_data["Count"].max(), 1)
        fig_sit = px.bar(
            _sit_data,
            x="Situation", y="Count",
            text="Count",
            color_discrete_sequence=["#004c54"],
        )
        fig_sit.update_traces(textposition="outside")
        fig_sit.update_layout(
            showlegend=False,
            xaxis_title="Score Situation",
            **CHART_LAYOUT,
        )
        fig_sit.update_yaxes(title="Count", range=[0, _max_sit * 1.3])
        st.plotly_chart(fig_sit, width="stretch")



# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"<div style='text-align:center;padding:12px 0 4px 0;'>"
    f"<p style='color:{MUTED};font-size:0.72rem;margin:0;'>"
    f"Gus &amp; Dave Sports Podcast — dashboard prototype &nbsp;·&nbsp; "
    f"Created by <strong style='color:{TEAL};'>Brennan Simpson</strong>"
    f"</p></div>",
    unsafe_allow_html=True,
)
