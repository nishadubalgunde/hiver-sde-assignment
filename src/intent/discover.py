import pandas as pd

INPUT = "data/processed/apple_conversations.csv"
OUTPUT = "data/processed/apple_customer_messages.csv"

print("Loading reconstructed conversations...")

df = pd.read_csv(INPUT)

# Keep only customer messages
customers = df[df["inbound"] == True].copy()

# Remove empty messages
customers["text"] = customers["text"].fillna("").str.strip()
customers = customers[customers["text"] != ""]

# Keep useful columns
customers = customers[
    [
        "conversation_id",
        "position",
        "tweet_id",
        "created_at",
        "text",
    ]
]

customers.to_csv(OUTPUT, index=False)

print()
print("=" * 60)
print("APPLE CUSTOMER MESSAGE DATASET")
print("=" * 60)

print(f"Input conversations: {df['conversation_id'].nunique():,}")
print(f"Customer messages: {len(customers):,}")
print(f"Output: {OUTPUT}")

print()
print("Sample customer messages:")

for _, row in customers.head(30).iterrows():
    print()
    print(f"Conversation: {row['conversation_id']}")
    print(f"Customer: {row['text']}")