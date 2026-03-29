import pandas as pd
import os

# -------------------------
# LOAD DATA
# -------------------------
matches = pd.read_csv("data/raw/matches.csv")
deliveries = pd.read_csv("data/raw/deliveries.csv")

# merge
df = deliveries.merge(matches, left_on="match_id", right_on="id")

print("After merge:", df.shape)

# -------------------------
# FILTERING
# -------------------------
df = df[df["inning"] == 2]
df = df[df["result"].isin(["runs", "wickets"])]
df = df[df["super_over"] == "N"]
df = df[df["method"].isna()]

print("After filtering:", df.shape)

# -------------------------
# SORT
# -------------------------
df = df.sort_values(["match_id", "over", "ball"])

# -------------------------
# TARGET
# -------------------------
df["win"] = (df["batting_team"] == df["winner"]).astype(int)

# -------------------------
# FEATURES
# -------------------------
df["current_score"] = df.groupby("match_id")["total_runs"].cumsum()

df["runs_remaining"] = df["target_runs"] - df["current_score"]

df["ball_number"] = df["over"] * 6 + df["ball"]
df["balls_remaining"] = 120 - df["ball_number"]

df["wickets_fallen"] = df.groupby("match_id")["is_wicket"].cumsum()
df["wickets_remaining"] = 10 - df["wickets_fallen"]

# -------------------------
# CLEANING (IMPORTANT ORDER)
# -------------------------
df = df[df["ball_number"] <= 120]
df = df[df["runs_remaining"] > 0]
df = df[df["balls_remaining"] > 0]   # FIXED (no zero)

# -------------------------
# RRR (AFTER CLEANING)
# -------------------------
df["rrr"] = df["runs_remaining"] / (df["balls_remaining"] / 6)

# remove any inf just in case
df = df.replace([float("inf"), -float("inf")], pd.NA)

print("After cleaning:", df.shape)

# -------------------------
# FINAL DATASET
# -------------------------
features = df[[
    "runs_remaining",
    "balls_remaining",
    "wickets_remaining",
    "rrr",
    "win"
]].dropna()

print("\nFINAL FEATURES:")
print(features.head())

print("\nFinal dataset shape:", features.shape)

# -------------------------
# SAVE
# -------------------------
os.makedirs("data/processed", exist_ok=True)
features.to_csv("data/processed/features.csv", index=False)

print("\nSaved to data/processed/features.csv")