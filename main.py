"""Predict NFL game margins from each team's season-to-date stats

Every feature is a team's average entering the game, so nothing predicts
itself. Early in a season those averages are pulled toward last year's, which
is what lets the model make a prediction in new seasons
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.linear_model import LinearRegression

CURRENT  = 2026
SEASONS  = list(range(2018, CURRENT + 1))     # pulls entire dataset
K        = 4                                  # games before this season outweighs last
BACKTEST = list(range(CURRENT - 5, CURRENT))  # walk-forward: train only on earlier seasons

STATS    = ["pf", "pa", "margin", "epa_play", "d_epa_play"]   # raw per-game measurements
SIDED    = ["pf", "pa", "epa_play", "d_epa_play", "sos"]      # per team, per game
FEATURES = [f"{side}_{c}" for side in ("home", "away") for c in SIDED]

# ---------- load ----------
raw = nfl.load_schedules(SEASONS).to_pandas()
raw = raw[raw["game_type"] == "REG"].reset_index(drop=True)
raw.dropna(subset=["home_score", "away_score"]).to_csv("games.csv", index=False)


def load_box_scores(seasons):
    """Weekly team stats. The current season has no file until a game is played."""
    frames = []
    for s in seasons:
        try:
            frames.append(nfl.load_team_stats([s], summary_level="week").to_pandas())
        except Exception:
            print(f"note: no team stats published for {s} yet")
    return pd.concat(frames, ignore_index=True)


ts = load_box_scores(sorted(raw.dropna(subset=["home_score"])["season"].unique()))
ts = ts[ts["season_type"] == "REG"]

# EPA per play on offense; the same number from the other sideline is defense
box = pd.DataFrame({
    "game_id":  ts["game_id"],
    "team":     ts["team"],
    "opp":      ts["opponent_team"],
    "epa_play": (ts["passing_epa"].fillna(0) + ts["rushing_epa"].fillna(0))
                / (ts["attempts"] + ts["carries"] + ts["sacks_suffered"]),
})
box = box.merge(box[["game_id", "team", "epa_play"]]
                   .rename(columns={"team": "opp", "epa_play": "d_epa_play"}),
                on=["game_id", "opp"])


# ---------- reusable sideways flip ----------
def to_long(g):
    """One row per team per game, played or not."""
    h = g[["game_id", "season", "week", "home_team", "away_team",
           "home_score", "away_score"]].rename(
        columns={"home_team": "team", "away_team": "opp",
                 "home_score": "pf", "away_score": "pa"})
    a = g[["game_id", "season", "week", "away_team", "home_team",
           "away_score", "home_score"]].rename(
        columns={"away_team": "team", "home_team": "opp",
                 "away_score": "pf", "home_score": "pa"})
    return pd.concat([h, a]).sort_values(["season", "week"]).reset_index(drop=True)


long = to_long(raw).merge(box.drop(columns="opp"), on=["game_id", "team"], how="left")
long["margin"] = long["pf"] - long["pa"]


# ---------- team strength entering each game ----------
def strength_entering(long):
    """Season-to-date averages, blended toward last season while n is small.

    Unplayed games contribute nothing and inherit the last state we knew, so
    the same table serves both training and next week's predictions.
    """
    g = long.groupby(["season", "team"], sort=False)

    after = g[STATS].transform(lambda s: s.expanding().mean())
    after["n"] = g.cumcount() + 1.0
    after.loc[long["pf"].isna()] = np.nan

    # shift(1) drops the current game; ffill carries through byes and future weeks
    entering = after.groupby([long["season"], long["team"]], sort=False).transform(
        lambda s: s.shift(1).ffill())
    n = entering.pop("n").fillna(0.0)

    last_year = (long.groupby(["season", "team"])[STATS].mean()
                     .rename(index=lambda s: s + 1, level="season"))
    league    = long.groupby("season")[STATS].mean().rename(index=lambda s: s + 1)

    idx  = pd.MultiIndex.from_arrays([long["season"], long["team"]])
    base = last_year.reindex(idx)
    base = base.fillna(league.reindex(long["season"]).set_axis(idx))  # relocated franchises
    base = base.fillna(long[STATS].mean())                           # 2018, no prior season
    base.index = long.index

    w   = (n / (n + K)).to_numpy()[:, None]    # 0 games -> all last season
    out = pd.DataFrame(w * entering[STATS].fillna(0.0).to_numpy()
                       + (1 - w) * base[STATS].to_numpy(),
                       index=long.index, columns=STATS)
    out[["game_id", "season", "week", "team", "opp"]] = \
        long[["game_id", "season", "week", "team", "opp"]]

    # strength of schedule: how good were the teams you already played?
    faced = out[["season", "week", "team", "margin"]].rename(
        columns={"team": "opp", "margin": "opp_margin"})
    out = out.merge(faced, on=["season", "week", "opp"], how="left")
    out["sos"] = (out.sort_values(["season", "week"])
                     .groupby(["season", "team"], sort=False)["opp_margin"]
                     .transform(lambda s: s.shift(1).expanding().mean())
                     .fillna(0.0))
    return out


def attach(games, strength):
    """Widen the per-team table back onto one row per game."""
    s = strength[["game_id", "team"] + SIDED]
    for side in ("home", "away"):
        ren = {c: f"{side}_{c}" for c in SIDED} | {"team": f"{side}_team"}
        games = games.merge(s.rename(columns=ren), on=["game_id", f"{side}_team"], how="left")
    return games


games = attach(raw.copy(), strength_entering(long))
games["y"] = games["home_score"] - games["away_score"]
fit = games.dropna(subset=FEATURES + ["y"])


# ---------- walk-forward backtest ----------
print(f"{'season':>6} {'games':>6} {'acc':>7} {'mae':>7} {'vegas':>7}")
hits, err, seen = 0.0, 0.0, 0
for s in BACKTEST:
    train, test = fit[fit["season"] < s], fit[fit["season"] == s]
    if not len(test):
        continue
    p   = LinearRegression().fit(train[FEATURES], train["y"]).predict(test[FEATURES])
    acc = ((p > 0) == (test["y"] > 0)).mean()
    mae = np.abs(p - test["y"]).mean()
    mkt = test.dropna(subset=["spread_line"])
    vg  = ((mkt["spread_line"] > 0) == (mkt["y"] > 0)).mean() if len(mkt) else np.nan
    hits, err, seen = hits + acc * len(test), err + mae * len(test), seen + len(test)
    print(f"{s:>6} {len(test):>6} {acc:>7.3f} {mae:>7.2f} {vg:>7.3f}")

if seen:
    print(f"{'all':>6} {seen:>6} {hits / seen:>7.3f} {err / seen:>7.2f}")
    print(f"(home team wins {(fit['y'] > 0).mean():.3f} of the time)")

# ---------- fit on everything, then call the next slate ----------
model = LinearRegression().fit(fit[FEATURES], fit["y"])
print(f"\ntrain rows: {len(fit)}")
for name, coef in zip(FEATURES, model.coef_):
    print(f"  {name:16} {coef:7.3f}")
print(f"  {'intercept':16} {model.intercept_:7.3f}")

upcoming = games[games["season"].eq(CURRENT) & games["home_score"].isna()]
upcoming = upcoming.dropna(subset=FEATURES)

if len(upcoming):
    week = upcoming[upcoming["week"] == upcoming["week"].min()].copy()
    week["pred"] = model.predict(week[FEATURES])
    print(f"\n--- Week {int(week['week'].iloc[0])} predictions ---")
    print(pd.DataFrame({
        "matchup": week["away_team"] + " @ " + week["home_team"],
        "pick":    np.where(week["pred"] > 0, week["home_team"], week["away_team"]),
        "margin":  week["pred"].abs().round(1),
        "vegas":   week["spread_line"],
        "edge":    (week["pred"] - week["spread_line"]).round(1),
    }).to_string(index=False))
else:
    print("\nNo unplayed games in the schedule.")
