"""
UFL Rule Impact Analysis on NFL Play-by-Play Data
==================================================
We are modeling rule impacts, not game theory responses.
Assume teams behave exactly as they did historically.
"""

import nfl_data_py as nfl

# ── Load data ──────────────────────────────────────────────────────────────────
pbp = nfl.import_pbp_data([2025], downcast=True)
pbp = pbp[pbp["season_type"] == "REG"]

# Work on views/copies only — original dataframe is never modified.

# ==============================================================================
# SECTION 1: PUNTS
# ==============================================================================

punts = pbp[pbp["play_type"] == "punt"]
total_punts = len(punts)

# yardline_100 < 50  →  punting team is past midfield (opponent's territory)
punts_in_opp_territory = punts[punts["yardline_100"] < 50]

# Exemption: rule does NOT apply in the final 2 minutes of either half.
# half_seconds_remaining counts down from 1800→0 across each half, so <=120
# captures the last 2 min before halftime AND the last 2 min before end of game.
IN_TWO_MIN_WINDOW = punts_in_opp_territory["half_seconds_remaining"] <= 120

punts_exempt  = punts_in_opp_territory[IN_TWO_MIN_WINDOW]   # allowed by exemption
punts_illegal = punts_in_opp_territory[~IN_TWO_MIN_WINDOW]  # truly banned

n_opp_territory   = len(punts_in_opp_territory)
n_exempt_punts    = len(punts_exempt)
n_illegal_punts   = len(punts_illegal)
pct_illegal_punts = n_illegal_punts / total_punts * 100 if total_punts > 0 else 0

print("=" * 60)
print("SECTION 1 — PUNTS (league-wide)")
print("=" * 60)
print(f"  Total punts (season):                          {total_punts:>6,}")
print(f"  Punts in opp territory (yd_100 < 50):          {n_opp_territory:>6,}")
print(f"    └─ Exempt (inside 2-min warning):            {n_exempt_punts:>6,}")
print(f"    └─ Truly illegal under UFL rule:             {n_illegal_punts:>6,}")
print(f"  % of all punts that would be banned:           {pct_illegal_punts:>6.1f}%")
print()

# ── Team-by-team punt breakdown ────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1a — PUNTS BY TEAM")
print("=" * 60)

team_punts = (
    punts
    .groupby("posteam")
    .agg(
        total_punts   =("play_type", "count"),
        illegal_punts =(
            "yardline_100",
            lambda x: ((x < 50) & (punts.loc[x.index, "half_seconds_remaining"] > 120)).sum()
        ),
    )
    .assign(pct_illegal=lambda df: (df["illegal_punts"] / df["total_punts"] * 100).round(1))
    .sort_values("illegal_punts", ascending=False)
)

print(team_punts.to_string())
print()

# ==============================================================================
# SECTION 2: FIELD GOALS — 60+ YARD RULE (4 pts instead of 3)
# ==============================================================================

fgs = pbp[pbp["play_type"] == "field_goal"]
total_fg_attempts = len(fgs)

long_fgs      = fgs[fgs["kick_distance"] >= 60]
n_long_fg_att = len(long_fgs)

made_long_fgs  = long_fgs[long_fgs["field_goal_result"] == "made"]
n_long_fg_made = len(made_long_fgs)

# Each made 60+ yarder is currently worth 3 pts; under UFL rules it's 4 → +1 per make.
extra_points = n_long_fg_made * 1

print("=" * 60)
print("SECTION 2 — FIELD GOALS (60+ yard = 4 pts, league-wide)")
print("=" * 60)
print(f"  Total FG attempts (season):              {total_fg_attempts:>6,}")
print(f"  FG attempts from 60+ yards:              {n_long_fg_att:>6,}")
print(f"  Made FGs from 60+ yards:                 {n_long_fg_made:>6,}")
print(f"  FG % from 60+ yards:                     "
      f"{n_long_fg_made/n_long_fg_att*100:>5.1f}%"
      if n_long_fg_att > 0 else "  FG % from 60+ yards:                        N/A")
print(f"  Extra points league-wide (+1 per make):  {extra_points:>6,}")
print()

# ── Team-by-team FG breakdown ──────────────────────────────────────────────────
print("=" * 60)
print("SECTION 2a — 60+ YARD FIELD GOALS BY TEAM")
print("=" * 60)

team_fgs = (
    long_fgs
    .groupby("posteam")
    .agg(
        fg_att_60plus =("field_goal_result", "count"),
        fg_made_60plus=("field_goal_result", lambda x: (x == "made").sum()),
    )
    .assign(
        make_pct  =lambda df: (df["fg_made_60plus"] / df["fg_att_60plus"] * 100).round(1),
        extra_pts =lambda df: df["fg_made_60plus"] * 1,
    )
    .sort_values("extra_pts", ascending=False)
)

print(team_fgs.to_string())
print()

# ==============================================================================
# SECTION 2b: DID THE TEAM LOSE BY 1 IN THE GAME THEY MADE A 60+ YARD FG?
# ==============================================================================

# Build a final-score lookup: max cumulative score per game = final score.
final_scores = (
    pbp
    .groupby("game_id")
    .agg(
        final_home_score =("total_home_score", "max"),
        final_away_score =("total_away_score", "max"),
    )
    .reset_index()
)

# Join final scores onto each made 60+ FG play.
# home_team is already present on made_long_fgs (inherited from pbp).
made_with_result = made_long_fgs.merge(
    final_scores[["game_id", "final_home_score", "final_away_score"]],
    on="game_id",
    how="left",
)

# Margin from the kicking team's perspective (negative = loss).
made_with_result = made_with_result.assign(
    team_margin=lambda df: (
        (df["final_home_score"] - df["final_away_score"])
        .where(df["posteam"] == df["home_team"],
               df["final_away_score"] - df["final_home_score"])
    )
)

lost_by_1 = made_with_result[made_with_result["team_margin"] == -1]

print("=" * 60)
print("SECTION 2b — MADE 60+ YD FGs WHERE TEAM LOST BY 1 PT")
print("=" * 60)
print(f"  Made 60+ yd FGs total:            {n_long_fg_made:>4,}")
print(f"  Games where kicker's team lost by 1: {len(lost_by_1):>4,}")
print()

if not lost_by_1.empty:
    cols = ["game_id", "posteam", "kick_distance", "final_home_score", "final_away_score", "team_margin"]
    print(lost_by_1[cols].to_string(index=False))
else:
    print("  No instances found.")
print()

# ==============================================================================
# SUMMARY
# ==============================================================================
print("=" * 60)
print("SUMMARY — UFL Rule Impact (historical behavior assumed)")
print("=" * 60)
print(f"  Punts BANNED under UFL rules:               {n_illegal_punts:,} "
      f"({pct_illegal_punts:.1f}% of all punts)")
print(f"  (excl. {n_exempt_punts} opp-territory punts inside 2-min warning)")
print(f"  Extra points from 4-pt long FG rule:        {extra_points:,} pts")
print(f"  (across {n_long_fg_made} made 60+ yd FGs out of {n_long_fg_att} attempts)")
print(f"  Made 60+ yd FGs where team lost by 1 pt:    {len(lost_by_1):,}")
