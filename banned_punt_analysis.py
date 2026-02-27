"""
Banned Punt Deep-Dive Analysis
===============================
We are modeling rule impacts, not game theory responses.
Assume teams behave exactly as they did historically.

This script analyzes punts that would be BANNED under the UFL rule:
  - play_type == "punt"
  - yardline_100 < 50  (punting team past midfield)
  - half_seconds_remaining > 120  (not inside 2-minute warning)
"""

import nfl_data_py as nfl
import pandas as pd

# ── Load data ──────────────────────────────────────────────────────────────────
pbp = nfl.import_pbp_data([2025], downcast=True)
pbp = pbp[pbp["season_type"] == "REG"]

# ── Isolate banned punts (same logic as ufl_rule_analysis.py) ─────────────────
punts             = pbp[pbp["play_type"] == "punt"]
punts_in_opp      = punts[punts["yardline_100"] < 50]
banned_punts      = punts_in_opp[punts_in_opp["half_seconds_remaining"] > 120]

n_banned = len(banned_punts)

print("=" * 65)
print(f"Banned punts identified: {n_banned:,}")
print("=" * 65)
print()

# ==============================================================================
# SECTION A: DISTRIBUTION OF 4TH-DOWN DISTANCE (ydstogo)
# ==============================================================================

ydstogo = banned_punts["ydstogo"]

print("=" * 65)
print("SECTION A — 4th-Down Distance on Banned Punts")
print("=" * 65)
print(f"  Mean yards to go:    {ydstogo.mean():.1f}")
print(f"  Median yards to go:  {ydstogo.median():.1f}")
print(f"  Std dev:             {ydstogo.std():.1f}")
print()

# ── Bucketed distribution ──────────────────────────────────────────────────────
BUCKET_LABELS = ["0–1", "2–3", "4–5", "6–10", "11+"]
BUCKET_BINS   = [0, 1, 3, 5, 10, float("inf")]

bucket_counts = (
    pd.cut(
        ydstogo,
        bins=BUCKET_BINS,
        labels=BUCKET_LABELS,
        right=True,
        include_lowest=True,
    )
    .value_counts()
    .reindex(BUCKET_LABELS)
)

bucket_pct = (bucket_counts / n_banned * 100).round(1)

bucket_table = pd.DataFrame({
    "count": bucket_counts,
    "pct"  : bucket_pct,
})

print("  ydstogo bucket distribution:")
print(bucket_table.to_string())
print()

# ==============================================================================
# SECTION B: BRACKET BREAKDOWN
# ==============================================================================

short   = banned_punts[banned_punts["ydstogo"] <= 2]
medium  = banned_punts[(banned_punts["ydstogo"] >= 3) & (banned_punts["ydstogo"] <= 5)]
long_   = banned_punts[banned_punts["ydstogo"] >= 6]

print("=" * 65)
print("SECTION B — Banned Punts by Distance Bracket")
print("=" * 65)
print(f"  4th-and-2 or less  (short):  {len(short):>4,}  ({len(short)/n_banned*100:.1f}%)")
print(f"  4th-and-3 to 5     (medium): {len(medium):>4,}  ({len(medium)/n_banned*100:.1f}%)")
print(f"  4th-and-6 or more  (long):   {len(long_):>4,}  ({len(long_)/n_banned*100:.1f}%)")
print()

# ==============================================================================
# SECTION C: EPA COMPARISON — BANNED PUNTS vs GOING FOR IT
# ==============================================================================

# Average EPA of the banned punt plays themselves.
avg_punt_epa = banned_punts["epa"].mean()

# All 4th-down plays where teams actually went for it (run or pass).
# This is the historical baseline for what going for it looks like.
fourth_go = pbp[
    (pbp["down"] == 4) &
    (pbp["play_type"].isin(["run", "pass"]))
]
avg_go_epa = fourth_go["epa"].mean()

print("=" * 65)
print("SECTION C — EPA: Banned Punts vs Going For It")
print("=" * 65)
print(f"  Avg EPA of banned punt plays:               {avg_punt_epa:>+.3f}")
print(f"  Avg EPA of all 4th-down go-for-it plays:   {avg_go_epa:>+.3f}")
print(f"  Raw EPA swing (go-for-it minus punt):       {avg_go_epa - avg_punt_epa:>+.3f}")
print()
print("  NOTE: The punt EPA is from the punting team's perspective.")
print("  A less-negative punt EPA means a better outcome for the kicker.")
print()

# ==============================================================================
# SECTION D: ESTIMATED EPA SWING IF TEAMS HAD GONE FOR IT
# ==============================================================================
# Assumptions (stated explicitly):
#   1. Conversion rates are derived from this same season's historical
#      4th-down go-for-it plays (play_type == run/pass), bucketed by ydstogo.
#   2. "Converted" is defined via fourth_down_converted == 1.
#   3. Expected EPA of going for it per bucket =
#        P(convert) * mean(EPA | converted) + P(fail) * mean(EPA | failed)
#   4. EPA swing = E[EPA going for it] - E[EPA punting] for that bucket.
#   5. We do NOT model opponent field position changes from a failed attempt;
#      EPA already encodes next-play value implicitly.

print("=" * 65)
print("SECTION D — Estimated EPA Swing If Teams Had Gone For It")
print("=" * 65)
print("  Assumptions:")
print("  - Conversion rates from this season's 4th-down go-for-it plays")
print("  - Bucketed by ydstogo (same buckets as above)")
print("  - Expected EPA = P(convert)*E[EPA|success] + P(fail)*E[EPA|fail]")
print("  - Modeling rule impact only; no adaptive strategy assumed")
print()

# Build historical conversion stats per bucket from actual go-for-it plays.
fourth_go_bucketed = fourth_go.copy()
fourth_go_bucketed["bucket"] = pd.cut(
    fourth_go_bucketed["ydstogo"],
    bins=BUCKET_BINS,
    labels=BUCKET_LABELS,
    right=True,
    include_lowest=True,
)

# Compute each stat separately to avoid pandas version compatibility issues
# with groupby.apply(include_groups=).
_base = fourth_go_bucketed.groupby("bucket", observed=True)

_attempts = _base["fourth_down_converted"].agg(
    go_attempts="count",
    conversions="sum",
    conv_rate="mean",
)

_epa_conv = (
    fourth_go_bucketed[fourth_go_bucketed["fourth_down_converted"] == 1]
    .groupby("bucket", observed=True)["epa"]
    .mean()
    .rename("epa_if_converted")
)

_epa_fail = (
    fourth_go_bucketed[fourth_go_bucketed["fourth_down_converted"] == 0]
    .groupby("bucket", observed=True)["epa"]
    .mean()
    .rename("epa_if_failed")
)

conversion_stats = (
    _attempts
    .join(_epa_conv)
    .join(_epa_fail)
    .reindex(BUCKET_LABELS)
)

conversion_stats["exp_epa_go"] = (
    conversion_stats["conv_rate"] * conversion_stats["epa_if_converted"]
    + (1 - conversion_stats["conv_rate"]) * conversion_stats["epa_if_failed"]
)

# Average EPA of the banned punts within each bucket.
banned_punts_bucketed = banned_punts.copy()
banned_punts_bucketed["bucket"] = pd.cut(
    banned_punts_bucketed["ydstogo"],
    bins=BUCKET_BINS,
    labels=BUCKET_LABELS,
    right=True,
    include_lowest=True,
)

punt_epa_by_bucket = (
    banned_punts_bucketed
    .groupby("bucket", observed=True)["epa"]
    .mean()
    .reindex(BUCKET_LABELS)
    .rename("avg_punt_epa")
)

results = conversion_stats.join(punt_epa_by_bucket)
results["epa_swing"] = results["exp_epa_go"] - results["avg_punt_epa"]

# Format for readability
display = results[[
    "go_attempts", "conv_rate", "epa_if_converted",
    "epa_if_failed", "exp_epa_go", "avg_punt_epa", "epa_swing"
]].copy()

display["conv_rate"]       = display["conv_rate"].map("{:.1%}".format)
display["epa_if_converted"] = display["epa_if_converted"].map("{:+.3f}".format)
display["epa_if_failed"]   = display["epa_if_failed"].map("{:+.3f}".format)
display["exp_epa_go"]      = display["exp_epa_go"].map("{:+.3f}".format)
display["avg_punt_epa"]    = display["avg_punt_epa"].map("{:+.3f}".format)
display["epa_swing"]       = display["epa_swing"].map("{:+.3f}".format)

print(display.to_string())
print()
print("  epa_swing > 0 → going for it expected to be better than punting")
print("  epa_swing < 0 → punting was actually the higher-EPA decision")
print()

# ==============================================================================
# SECTION E: WIN PROBABILITY SWING IF TEAMS HAD GONE FOR IT
# ==============================================================================
# Assumptions (stated explicitly):
#   1. wp  = pre-play win probability for the possession team (from pbp).
#   2. Post-play WP = wp + wpa (win probability after the play resolves).
#   3. For punts: we measure avg post-punt WP (wp + wpa) per bucket.
#   4. For go-for-it: E[post-play WP] per bucket =
#        P(convert) * mean(wp+wpa | converted) + P(fail) * mean(wp+wpa | failed)
#      using the same historical 4th-down go-for-it plays as Section D.
#   5. WP swing = E[WP go-for-it] - avg post-punt WP.
#      Positive swing → going for it preserves more win probability.

print("=" * 65)
print("SECTION E — Win Probability Swing If Teams Had Gone For It")
print("=" * 65)
print("  Assumptions:")
print("  - wp/wpa from pbp are possession-team win probability")
print("  - Post-play WP = wp + wpa")
print("  - Same conversion rates and buckets as Section D")
print("  - Modeling rule impact only; no adaptive strategy assumed")
print()

# ── Headline: overall avg pre-play WP and post-punt WP ────────────────────────
avg_wp_before_punt  = banned_punts["wp"].mean()
avg_wp_after_punt   = (banned_punts["wp"] + banned_punts["wpa"]).mean()
avg_wp_go_overall   = (
    fourth_go["fourth_down_converted"].mean() *
        (fourth_go.loc[fourth_go["fourth_down_converted"] == 1, "wp"] +
         fourth_go.loc[fourth_go["fourth_down_converted"] == 1, "wpa"]).mean()
    + (1 - fourth_go["fourth_down_converted"].mean()) *
        (fourth_go.loc[fourth_go["fourth_down_converted"] == 0, "wp"] +
         fourth_go.loc[fourth_go["fourth_down_converted"] == 0, "wpa"]).mean()
)

print(f"  Avg WP before banned punt:               {avg_wp_before_punt:>6.3f}")
print(f"  Avg WP after punt (wp + wpa):            {avg_wp_after_punt:>6.3f}")
print(f"  Avg expected WP if gone for it:          {avg_wp_go_overall:>6.3f}")
print(f"  Overall WP swing (go-for-it vs punt):   {avg_wp_go_overall - avg_wp_after_punt:>+6.3f}")
print()

# ── Bucket-level WP swing ──────────────────────────────────────────────────────
# Post-play WP for historical go-for-it plays.
fourth_go_bucketed["post_play_wp"] = fourth_go_bucketed["wp"] + fourth_go_bucketed["wpa"]

_wp_conv = (
    fourth_go_bucketed[fourth_go_bucketed["fourth_down_converted"] == 1]
    .groupby("bucket", observed=True)["post_play_wp"]
    .mean()
    .rename("wp_if_converted")
)

_wp_fail = (
    fourth_go_bucketed[fourth_go_bucketed["fourth_down_converted"] == 0]
    .groupby("bucket", observed=True)["post_play_wp"]
    .mean()
    .rename("wp_if_failed")
)

# Post-punt WP for each banned punt bucket.
banned_punts_bucketed["post_punt_wp"] = (
    banned_punts_bucketed["wp"] + banned_punts_bucketed["wpa"]
)

punt_wp_by_bucket = (
    banned_punts_bucketed
    .groupby("bucket", observed=True)["post_punt_wp"]
    .mean()
    .reindex(BUCKET_LABELS)
    .rename("avg_post_punt_wp")
)

# Reuse conv_rate from Section D conversion_stats.
wp_results = (
    conversion_stats[["conv_rate"]]
    .join(_wp_conv)
    .join(_wp_fail)
    .join(punt_wp_by_bucket)
    .reindex(BUCKET_LABELS)
)

wp_results["exp_wp_go"] = (
    wp_results["conv_rate"] * wp_results["wp_if_converted"]
    + (1 - wp_results["conv_rate"]) * wp_results["wp_if_failed"]
)

wp_results["wp_swing"] = wp_results["exp_wp_go"] - wp_results["avg_post_punt_wp"]

# Format for readability
wp_display = wp_results[[
    "conv_rate", "wp_if_converted", "wp_if_failed",
    "exp_wp_go", "avg_post_punt_wp", "wp_swing"
]].copy()

wp_display["conv_rate"]       = wp_display["conv_rate"].map("{:.1%}".format)
wp_display["wp_if_converted"] = wp_display["wp_if_converted"].map("{:.3f}".format)
wp_display["wp_if_failed"]    = wp_display["wp_if_failed"].map("{:.3f}".format)
wp_display["exp_wp_go"]       = wp_display["exp_wp_go"].map("{:.3f}".format)
wp_display["avg_post_punt_wp"]= wp_display["avg_post_punt_wp"].map("{:.3f}".format)
wp_display["wp_swing"]        = wp_display["wp_swing"].map("{:+.3f}".format)

print(wp_display.to_string())
print()
print("  wp_swing > 0 → going for it preserves more win probability than punting")
print("  wp_swing < 0 → punting was the higher win-probability play in that bucket")
print()

# ==============================================================================
# SECTION F: BANNED PUNTS BY SCORE DIFFERENTIAL AT TIME OF PUNT
# ==============================================================================
# score_differential = possession team score minus opponent score.
# Trailing (1 score)  = down 1–8 pts  (a TD+2pt conversion closes the gap)
# Trailing (2+ scores) = down 9+ pts

def score_situation(diff):
    if diff > 0:
        return "Leading"
    elif diff == 0:
        return "Tied"
    elif diff >= -8:
        return "Trailing (1 score)"
    else:
        return "Trailing (2+ scores)"

SITUATION_ORDER = ["Leading", "Tied", "Trailing (1 score)", "Trailing (2+ scores)"]

banned_punts_scored = banned_punts.assign(
    situation=banned_punts["score_differential"].apply(score_situation)
)

situation_stats = (
    banned_punts_scored
    .groupby("situation")
    .agg(
        count      =("ydstogo", "count"),
        avg_ydstogo=("ydstogo", "mean"),
    )
    .reindex(SITUATION_ORDER)
)

situation_stats["pct"]         = (situation_stats["count"] / n_banned * 100).round(1)
situation_stats["avg_ydstogo"] = situation_stats["avg_ydstogo"].round(1)

print("=" * 65)
print("SECTION F — Banned Punts by Score Situation at Time of Punt")
print("=" * 65)
print(f"  {'Situation':<24} {'Count':>6}  {'%':>6}  {'Avg ydstogo':>12}")
print(f"  {'-'*24} {'-'*6}  {'-'*6}  {'-'*12}")
for situation, row in situation_stats.iterrows():
    if pd.isna(row["count"]):
        continue
    print(f"  {situation:<24} {int(row['count']):>6}  {row['pct']:>5.1f}%  {row['avg_ydstogo']:>12.1f}")
print()

# ==============================================================================
# SECTION G: WP SWING — LEADING TEAMS ONLY
# ==============================================================================
# Focus: banned punts where score_differential > 0 (team was ahead).
# WP reference for go-for-it outcomes is also filtered to leading situations
# so that the post-play WP values reflect a comparable game state.

leading_banned = banned_punts[banned_punts["score_differential"] > 0].copy()
n_leading      = len(leading_banned)

# Historical go-for-it plays when team was also leading.
fourth_go_leading = fourth_go[fourth_go["score_differential"] > 0].copy()

print("=" * 65)
print("SECTION G — WP Swing on Banned Punts When Team Was LEADING")
print("=" * 65)
print(f"  Banned punts while leading:  {n_leading:,} of {n_banned:,} total "
      f"({n_leading/n_banned*100:.1f}%)")
print()
print("  Assumptions:")
print("  - Filtered to score_differential > 0 at time of punt")
print("  - Go-for-it WP reference also filtered to leading situations")
print("  - Same ydstogo buckets and conversion modeling as Section D/E")
print()

# ── Headline numbers ───────────────────────────────────────────────────────────
avg_wp_before  = leading_banned["wp"].mean()
avg_wp_after   = (leading_banned["wp"] + leading_banned["wpa"]).mean()

# Expected WP if gone for it (overall, not bucketed) — weighted by conv rate
# derived from leading go-for-it plays.
_lead_conv_rate = fourth_go_leading["fourth_down_converted"].mean()
_lead_wp_conv   = (
    fourth_go_leading.loc[fourth_go_leading["fourth_down_converted"] == 1, "wp"] +
    fourth_go_leading.loc[fourth_go_leading["fourth_down_converted"] == 1, "wpa"]
).mean()
_lead_wp_fail   = (
    fourth_go_leading.loc[fourth_go_leading["fourth_down_converted"] == 0, "wp"] +
    fourth_go_leading.loc[fourth_go_leading["fourth_down_converted"] == 0, "wpa"]
).mean()

avg_wp_go_leading = _lead_conv_rate * _lead_wp_conv + (1 - _lead_conv_rate) * _lead_wp_fail

print(f"  Avg WP before punt (while leading):      {avg_wp_before:>6.3f}")
print(f"  Avg WP after punt  (wp + wpa):           {avg_wp_after:>6.3f}")
print(f"  Avg expected WP if gone for it:          {avg_wp_go_leading:>6.3f}")
print(f"  Overall WP swing (go-for-it vs punt):   {avg_wp_go_leading - avg_wp_after:>+6.3f}")
print()

# ── Bucket-level breakdown ─────────────────────────────────────────────────────
leading_banned["bucket"] = pd.cut(
    leading_banned["ydstogo"],
    bins=BUCKET_BINS, labels=BUCKET_LABELS, right=True, include_lowest=True,
)
fourth_go_leading["bucket"] = pd.cut(
    fourth_go_leading["ydstogo"],
    bins=BUCKET_BINS, labels=BUCKET_LABELS, right=True, include_lowest=True,
)
fourth_go_leading["post_play_wp"] = fourth_go_leading["wp"] + fourth_go_leading["wpa"]

# Conversion rate per bucket (from leading go-for-it plays).
_lead_attempts = (
    fourth_go_leading
    .groupby("bucket", observed=True)["fourth_down_converted"]
    .agg(go_attempts="count", conv_rate="mean")
)

# Post-play WP split by outcome (from leading go-for-it plays).
_lead_wp_c = (
    fourth_go_leading[fourth_go_leading["fourth_down_converted"] == 1]
    .groupby("bucket", observed=True)["post_play_wp"]
    .mean()
    .rename("wp_if_converted")
)
_lead_wp_f = (
    fourth_go_leading[fourth_go_leading["fourth_down_converted"] == 0]
    .groupby("bucket", observed=True)["post_play_wp"]
    .mean()
    .rename("wp_if_failed")
)

# Post-punt WP for leading banned punts per bucket.
_lead_punt_wp = (
    leading_banned
    .assign(post_punt_wp=leading_banned["wp"] + leading_banned["wpa"])
    .groupby("bucket", observed=True)
    .agg(punt_count=("post_punt_wp", "count"), avg_post_punt_wp=("post_punt_wp", "mean"))
    .reindex(BUCKET_LABELS)
)

g_results = (
    _lead_attempts
    .join(_lead_wp_c)
    .join(_lead_wp_f)
    .join(_lead_punt_wp)
    .reindex(BUCKET_LABELS)
)

g_results["exp_wp_go"] = (
    g_results["conv_rate"] * g_results["wp_if_converted"]
    + (1 - g_results["conv_rate"]) * g_results["wp_if_failed"]
)
g_results["wp_swing"] = g_results["exp_wp_go"] - g_results["avg_post_punt_wp"]

g_display = g_results[[
    "punt_count", "conv_rate", "wp_if_converted",
    "wp_if_failed", "exp_wp_go", "avg_post_punt_wp", "wp_swing"
]].copy()

g_display["conv_rate"]        = g_display["conv_rate"].map("{:.1%}".format)
g_display["wp_if_converted"]  = g_display["wp_if_converted"].map("{:.3f}".format)
g_display["wp_if_failed"]     = g_display["wp_if_failed"].map("{:.3f}".format)
g_display["exp_wp_go"]        = g_display["exp_wp_go"].map("{:.3f}".format)
g_display["avg_post_punt_wp"] = g_display["avg_post_punt_wp"].map("{:.3f}".format)
g_display["wp_swing"]         = g_display["wp_swing"].map("{:+.3f}".format)

print(g_display.to_string())
print()
print("  wp_swing > 0 → going for it preserves more WP even while leading")
print("  wp_swing < 0 → punting was the correct WP-maximizing play while leading")
