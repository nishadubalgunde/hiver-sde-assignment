import pandas as pd

PATH = "data/processed/apple_support.csv"

df = pd.read_csv(PATH)

print("=" * 60)
print("APPLE SUPPORT DATASET INSPECTION")
print("=" * 60)

print(f"\nTotal rows: {len(df):,}")

print("\nColumns:")
print(df.columns.tolist())

print("\nInbound distribution:")
print(df["inbound"].value_counts())

print("\nInbound percentages:")
print(df["inbound"].value_counts(normalize=True).mul(100).round(2))

print(f"\nUnique authors: {df['author_id'].nunique():,}")

print(f"Unique tweets: {df['tweet_id'].nunique():,}")

print(f"Missing text: {df['text'].isna().sum():,}")

print(f"Empty text: {(df['text'].fillna('').str.strip() == '').sum():,}")

print(
    f"Messages with parent tweet: "
    f"{df['in_response_to_tweet_id'].notna().sum():,}"
)

print(
    f"Messages with response tweet: "
    f"{df['response_tweet_id'].notna().sum():,}"
)

print("\nDate range:")
print("Start:", df["created_at"].min())
print("End:", df["created_at"].max())

print("\n" + "=" * 60)
print("SAMPLE CUSTOMER MESSAGES")
print("=" * 60)

customer = df[df["inbound"] == True]

for _, row in customer.head(10).iterrows():
    print(f"\nTweet ID: {row['tweet_id']}")
    print(f"Author: {row['author_id']}")
    print(f"Text: {row['text']}")

print("\n" + "=" * 60)
print("SAMPLE APPLE SUPPORT REPLIES")
print("=" * 60)

support = df[df["inbound"] == False]

for _, row in support.head(10).iterrows():
    print(f"\nTweet ID: {row['tweet_id']}")
    print(f"Text: {row['text']}")