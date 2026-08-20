import nflreadpy as nfl
import pandas as pd
from sklearn.linear_model import LinearRegression

FEATURES = ["home_pf", "home_pa", "away_pf", "away_pa"]

games = nfl.load_schedules([2018, 2024]).to_pandas()
games = games[games["game_type"] == "REG"]
games = games.dropna(subset=["home_score", "away_score"])

games.to_csv("games.csv", index=False)

# Flip table sideways so each row is one team per game
home = games[["season","week","home_team","home_score","away_score"]].rename(
    columns={"home_team":"team", "home_score":"pf", "away_score":"pa"})
away = games[["season","week","away_team","away_score","home_score"]].rename(
    columns={"away_team":"team", "away_score":"pf", "home_score":"pa"})

long = pd.concat([home, away]).sort_values(["season","week"])
long["margin"] = long["pf"] - long["pa"]

# Prevent stats from including the current week
# shift(1) pushes everything down one row so current game drops out
for col in ["margin", "pf", "pa"]:
    long[f"prior_{col}"] = (
        long.groupby(["season", "team"])[col]
            .transform(lambda s: s.shift(1).expanding().mean())
    )

# Trim to just the join key and the values we want
lookup = long[["season", "week", "team", "prior_margin", "prior_pf", "prior_pa"]]

# Join once matching the home team
games = games.merge(
    lookup.rename(columns={
        "team": "home_team",
        "prior_margin": "home_margin",
        "prior_pf": "home_pf",
        "prior_pa": "home_pa"}),
    on=["season", "week", "home_team"],
    how="left"
)

# Join again matching the away team
games = games.merge(
    lookup.rename(columns={
        "team": "away_team",
        "prior_margin": "away_margin",
        "prior_pf": "away_pf",
        "prior_pa": "away_pa"}),
    on=["season", "week", "away_team"],
    how="left"
)

# prior_* columns are each team's season averages heading into the current week
# print(games[["season","week","home_team","away_team",
            # "home_pf","home_pa","away_pf","away_pa"]].head(20))

df = games.dropna(subset=FEATURES)
# drop early weeks where averages are built on too few games
df = df[df["week"] > 4]

train = df[df["season"] < 2024]
test  = df[df["season"] == 2024]

y_train = train["home_score"] - train["away_score"]
model = LinearRegression().fit(train[FEATURES], y_train)

for name, coef in zip(FEATURES, model.coef_):
    print(f"{name:12} {coef:7.3f}")
print(f"{'intercept':12} {model.intercept_:7.3f}")

# test model
preds = model.predict(test[FEATURES])
actual = test["home_score"] - test["away_score"]

print("avg error in points:", abs(preds - actual).mean())
print("winner accuracy:", ((preds > 0) == (actual > 0)).mean())
print("home team wins:", (actual > 0).mean())

train_preds = model.predict(train[FEATURES])
train_actual = train["home_score"] - train["away_score"]
print("train accuracy:", ((train_preds > 0) == (train_actual > 0)).mean())