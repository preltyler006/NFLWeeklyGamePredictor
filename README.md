# NFL Weekly Game Predictor

Predicts the final margin of upcoming NFL games from each team's season-to-date
performance, and grades itself against the Vegas line.

A single ~170-line script: it pulls public NFL data, builds a team-strength
rating that works even in Week 1, fits a linear model on eight seasons of
games, and prints picks for the next unplayed slate.

## Results

Walk-forward backtest: each season is predicted by a model trained only on
earlier seasons, so nothing is scored on data it was fit to.

| season | games | accuracy | MAE | Vegas accuracy |
|-------:|------:|---------:|----:|---------------:|
| 2021 | 272 | .636 | 11.19 | .621 |
| 2022 | 271 | .657 | 9.05 | .664 |
| 2023 | 272 | .647 | 10.34 | .680 |
| 2024 | 272 | .702 | 9.96 | .713 |
| 2025 | 272 | .640 | 10.40 | .654 |
| **all** | **1359** | **.656** | **10.19** | n/a |

Baseline: the home team wins **.539** of the time. So the model is doing real
work, and lands within about a point of the closing line, but it does not beat
Vegas, and beating Vegas is not a realistic goal for a model this size.

*(Figures from a run during the 2026 season, before Week 4. They shift slightly
as new games land.)*

## Quickstart

```bash
pip install -r python_requirements.txt
python main.py
```

No API keys, no config. Data comes from
[nflreadpy](https://github.com/nflverse/nflreadpy) over the network on each run,
so the first run is the slow one.

Output is three blocks: the backtest table above, the fitted coefficients, and
the picks:

```
--- Week 4 predictions ---
  matchup pick  margin  vegas  edge
PIT @ CLE  PIT     1.5   -2.5   1.0
NYJ @ CHI  CHI     9.9    3.5   6.4
 NE @ BUF  BUF     6.5    7.0  -0.5
```

- **pick / margin**: who the model likes, and by how much
- **vegas**: the spread from the home team's perspective (positive = home favored)
- **edge**: predicted home margin minus the line

A large `edge` is more likely to mean the model is missing something the market
knows than that you have found value. See [Limitations](#limitations).

## How it works

**1 · Load.** Regular-season schedules 2018–present, plus weekly team box
scores. Unplayed games come along with blank scores, and that is what gets
predicted at the end. Played games are also snapshotted to `games.csv`.

**2 · Build per-team rows.** Each game is split into two rows, one per sideline,
so "how good is this team" becomes a question you can answer with a running
average. Offensive EPA per play comes from the box scores; a team's *defensive*
EPA is just their opponent's offensive EPA in the same game.

**3 · Rate each team entering each game.** Season-to-date averages of points
for, points against, margin, and EPA on both sides of the ball. Every one of
them is shifted so it reflects only games already played.

The Week 1 problem (zero games played, no average to take) is handled by
shrinking toward last season:

```
w = n / (n + K)
rating = w * (this season) + (1 - w) * (last season)
```

With `K = 4`: zero games in means the rating is entirely last season's, four
games in it is an even split, and by late season it is mostly current form.
Teams with no prior season fall back to the league average, then to the global
mean.

Strength of schedule is layered on top: the average quality of opponents
already faced.

**4 · Fit and predict.** Ten features (five ratings × home/away) into a linear
regression on home margin. Simple enough that the printed coefficients are
readable: EPA per play dominates, and the intercept comes out around +12, which
is home-field advantage plus the scale of the EPA terms.

### Not predicting the present

Every rating is a team's average *entering* the game. This is the constraint the
middle of the script is built around: if a season average included the game
being predicted, the backtest would look excellent and the Friday picks would be
worthless. Each running average is shifted by one game before it is used
anywhere.

The same discipline makes future games work for free. An unplayed game
contributes nothing to the averages and inherits the last known rating, so next
week's matchups flow through the identical code path.

## Configuration

Four constants at the top of [`main.py`](main.py):

| name | default | meaning |
|---|---|---|
| `CURRENT` | `2026` | current season; everything else derives from it |
| `SEASONS` | `2018…CURRENT` | seasons loaded |
| `K` | `4` | games before current form outweighs last season |
| `BACKTEST` | `CURRENT-5 … CURRENT-1` | seasons held out and scored |

`BACKTEST` is a rolling five-year window. For a growing window that keeps every
season since 2021, use `list(range(2021, CURRENT))` instead.

## Limitations

- **No injuries, QB changes, rest, weather, or travel.** Several of these sit
  unused in the downloaded schedule (`temp`, `wind`, `home_rest`,
  `home_qb_name`). A backup quarterback starting is invisible to the model.
- **Linear.** Effects are assumed to add up independently, so interactions
  (a strong run defense mattering more against a run-heavy offense) cannot be
  expressed.
- **Ratings are unregularized season averages.** A 3-0 team looks genuinely
  elite beyond what `K` shrinks away.
- **Not a betting tool.** It is within about a point of the market's hit rate,
  which after vig is a losing proposition.


## Requirements

Python 3.10+ (the script uses the `|` dict merge operator), plus
`nflreadpy`, `pandas`, `numpy`, and `scikit-learn`. See
[`python_requirements.txt`](python_requirements.txt).
