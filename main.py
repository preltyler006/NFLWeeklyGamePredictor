import nflreadpy as nfl
import pandas as pd
from sklearn.linear_model import LinearRegression

FEATURES = ["home_pf", "home_pa", "away_pf", "away_pa"]
CURRENT  = 2026
SEASONS  = list(range(2018, CURRENT + 1))   # derived, so it can't drift
K        = 4        # games before this season outweighs last
EVALUATE = True     # False once you trust it, to train on everything

# ---------- load ----------
raw = nfl.load_schedules(SEASONS).to_pandas()
raw = raw[raw["game_type"] == "REG"]

games = raw.dropna(subset=["home_score", "away_score"]).copy()
games.to_csv("games.csv", index=False)

# ---------- reusable sideways flip ---------- 
def to_long(g):
    """One row per team per game."""
    h = g[["season","week","home_team","home_score","away_score"]].rename(
        columns={"home_team":"team", "home_score":"pf", "away_score":"pa"})
    a = g[["season","week","away_team","away_score","home_score"]].rename(
        columns={"away_team":"team", "away_score":"pf", "home_score":"pa"})
    return pd.concat([h, a]).sort_values(["season","week"])

long = to_long(games)
long["margin"] = long["pf"] - long["pa"]

# shift(1) drops the current game so it can't predict itself
for col in ["margin", "pf", "pa"]:
    long[f"prior_{col}"] = (
        long.groupby(["season", "team"])[col]
            .transform(lambda s: s.shift(1).expanding().mean())
    )

lookup = long[["season","week","team","prior_margin","prior_pf","prior_pa"]]

games = games.merge(
    lookup.rename(columns={"team":"home_team", "prior_margin":"home_margin",
                           "prior_pf":"home_pf", "prior_pa":"home_pa"}),
    on=["season","week","home_team"], how="left")

games = games.merge(
    lookup.rename(columns={"team":"away_team", "prior_margin":"away_margin",
                           "prior_pf":"away_pf", "prior_pa":"away_pa"}),
    on=["season","week","away_team"], how="left")

df = games.dropna(subset=FEATURES)
df = df[df["week"] > 4]          # early-season averages are too noisy

# ---------- fit ----------
if EVALUATE:
    train = df[df["season"] < 2024]
    test  = df[df["season"] == 2024]
else:
    train, test = df, None

y_train = train["home_score"] - train["away_score"]
model = LinearRegression().fit(train[FEATURES], y_train)

print(f"train rows: {len(train)}")
for name, coef in zip(FEATURES, model.coef_):
    print(f"{name:12} {coef:7.3f}")
print(f"{'intercept':12} {model.intercept_:7.3f}")

if test is not None and len(test):
    preds  = model.predict(test[FEATURES])
    actual = test["home_score"] - test["away_score"]
    print("\navg error in points:", round(abs(preds - actual).mean(), 2))
    print("winner accuracy:", round(((preds > 0) == (actual > 0)).mean(), 3))
    print("home team wins:", round((actual > 0).mean(), 3))

    tp = model.predict(train[FEATURES])
    ta = train["home_score"] - train["away_score"]
    print("train accuracy:", round(((tp > 0) == (ta > 0)).mean(), 3))

# ---------- predict upcoming ----------
def team_averages(g):
    """Season-to-date pf/pa per team, plus games played."""
    both = to_long(g)
    out = both.groupby("team")[["pf","pa"]].mean()
    out["n"] = both.groupby("team").size()
    return out

sched    = raw[raw["season"] == CURRENT]
played   = sched.dropna(subset=["home_score","away_score"])
upcoming = sched[sched["home_score"].isna()]

prev = team_averages(games[games["season"] == CURRENT - 1])

if len(played):
    cur = team_averages(played).reindex(prev.index)
    n = cur["n"].fillna(0)
    w = n / (n + K)                      # 0 games -> all last season
    strength = pd.DataFrame({
        "pf": w * cur["pf"].fillna(0) + (1 - w) * prev["pf"],
        "pa": w * cur["pa"].fillna(0) + (1 - w) * prev["pa"],
    })
else:
    strength = prev[["pf","pa"]]

def predict_week(week):
    rows = []
    for _, g in upcoming[upcoming["week"] == week].iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in strength.index or a not in strength.index:
            continue
        x = pd.DataFrame([[strength.loc[h,"pf"], strength.loc[h,"pa"],
                           strength.loc[a,"pf"], strength.loc[a,"pa"]]],
                         columns=FEATURES)
        m = model.predict(x)[0]
        rows.append({"matchup": f"{a} @ {h}",
                     "pick": h if m > 0 else a,
                     "margin": round(abs(m), 1)})
    return pd.DataFrame(rows)

if len(upcoming):
    wk = int(upcoming["week"].min())
    print(f"\n--- Week {wk} predictions ---")
    print(predict_week(wk).to_string(index=False))
else:
    print("\nNo unplayed games in the schedule.")