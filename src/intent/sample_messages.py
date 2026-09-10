import pandas as pd

INPUT = "data/processed/apple_customer_messages.csv"

df = pd.read_csv(INPUT)

print("=" * 70)
print("APPLE SUPPORT CUSTOMER MESSAGE SAMPLING")
print("=" * 70)

print(f"Total customer messages: {len(df):,}")

# Reproducible random sample
sample = df.sample(
    n=200,
    random_state=42
)

sample = sample.sort_values("conversation_id")

print()
print("200 RANDOM CUSTOMER MESSAGES")
print("=" * 70)

for i, (_, row) in enumerate(sample.iterrows(), start=1):
    print()
    print(f"[{i}] Conversation: {row['conversation_id']}")
    print(f"Message: {row['text']}")

# Save sample for inspection
OUTPUT = "data/processed/apple_customer_sample_200.csv"
sample.to_csv(OUTPUT, index=False)

print()
print("=" * 70)
print(f"Saved sample to: {OUTPUT}")
print("=" * 70)