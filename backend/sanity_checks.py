import pandas as pd

matches = pd.read_csv("data/raw/matches.csv")
deliveries = pd.read_csv("data/raw/deliveries.csv")

df = deliveries.merge(matches, left_on="match_id", right_on="id")

# -------------------------
# SAME FILTERS AS PREPROCESS
# -------------------------
df = df[df["inning"] == 2]
df = df[df["result"].isin(["runs", "wickets"])]
df = df[df["super_over"] == "N"]
df = df[df["method"].isna()]

df = df.sort_values(["match_id", "over", "ball"])

# features
df["current_score"] = df.groupby("match_id")["total_runs"].cumsum()
df["runs_remaining"] = df["target_runs"] - df["current_score"]
df["ball_number"] = df["over"] * 6 + df["ball"]
df["balls_remaining"] = 120 - df["ball_number"]

# -------------------------
# CLEANING (ADD THIS!!)
# -------------------------
df = df[df["ball_number"] <= 120]
df = df[df["runs_remaining"] > 0]
df = df[df["balls_remaining"] >= 0]

# -------------------------
# SANITY CHECKS
# -------------------------

print("Checking balls_remaining range:")
print(df["balls_remaining"].describe())

print("\nChecking runs_remaining range:")
print(df["runs_remaining"].describe())

print("\nNegative runs_remaining:", (df["runs_remaining"] < 0).sum())
print("Negative balls_remaining:", (df["balls_remaining"] < 0).sum())

# sample match
match_id = df["match_id"].iloc[0]
sample = df[df["match_id"] == match_id]

print("\nSample match progression:")
print(sample[["over", "ball", "runs_remaining", "balls_remaining"]].head(20))