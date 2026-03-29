import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# -------------------------
# LOAD CLEAN DATA
# -------------------------
data = pd.read_csv("data/processed/features.csv")

print("Loaded dataset:", data.shape)

# -------------------------
# SPLIT
# -------------------------
X = data[[
    "runs_remaining",
    "balls_remaining",
    "wickets_remaining",
    "rrr"
]]

y = data["win"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# TRAIN
# -------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------
# EVALUATE
# -------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# -------------------------
# SAMPLE PREDICTION
# -------------------------
sample = pd.DataFrame([[50, 30, 5, 10.0]], columns=[
    "runs_remaining", "balls_remaining", "wickets_remaining", "rrr"
])

prob = model.predict_proba(sample)[0][1]

print("\nSample situation: 50 runs, 30 balls, 5 wickets")
print("Win probability:", prob)