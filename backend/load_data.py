import pandas as pd

matches = pd.read_csv("data/raw/matches.csv")
deliveries = pd.read_csv("data/raw/deliveries.csv")

# dimensions (row/columbs)
print("Matches shape:", matches.shape)
print("Deliveries shape:", deliveries.shape)

print("\nMatches columns:")
print(matches.columns)

print("\nDeliveries columns:")
print(deliveries.columns)