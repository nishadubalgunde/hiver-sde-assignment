import pandas as pd

INPUT = r"C:\Users\Nisha\.cache\kagglehub\datasets\thoughtvector\customer-support-on-twitter\versions\10\twcs\twcs.csv"
OUTPUT = "data/processed/apple_support_all.csv"

print("Reading dataset...")

apple_tweet_ids = set()

# ---------------------------------------------------------
# STEP 1: Find all AppleSupport tweets
# ---------------------------------------------------------

for chunk in pd.read_csv(
    INPUT,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ],
    chunksize=100000,
):

    apple_rows = chunk[chunk["author_id"] == "AppleSupport"]

    apple_tweet_ids.update(
        apple_rows["tweet_id"].astype(int).tolist()
    )

print(f"AppleSupport tweets found: {len(apple_tweet_ids):,}")


# ---------------------------------------------------------
# STEP 2: Find tweets directly connected to AppleSupport
# ---------------------------------------------------------

related_chunks = []

for chunk in pd.read_csv(
    INPUT,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ],
    chunksize=100000,
):

    # Tweets that directly reply to an AppleSupport tweet
    parent_match = chunk[
        chunk["in_response_to_tweet_id"].isin(apple_tweet_ids)
    ]

    if not parent_match.empty:
        related_chunks.append(parent_match)

    # AppleSupport tweets themselves
    apple_match = chunk[
        chunk["author_id"] == "AppleSupport"
    ]

    if not apple_match.empty:
        related_chunks.append(apple_match)


# ---------------------------------------------------------
# STEP 3: Combine and remove duplicates
# ---------------------------------------------------------

result = pd.concat(related_chunks, ignore_index=True)

result = result.drop_duplicates(
    subset=["tweet_id"]
)

result = result.sort_values(
    "tweet_id"
)

result.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 60)
print("APPLE SUPPORT THREAD DATA CREATED")
print("=" * 60)

print(f"Output: {OUTPUT}")
print(f"Rows: {len(result):,}")

print()
print("Inbound distribution:")

print(
    result["inbound"]
    .value_counts()
)

print()
print("Unique authors:")
print(
    result["author_id"].nunique()
)

print()
print("Customer messages:")

print(
    len(result[result["inbound"] == True])
)

print()
print("AppleSupport messages:")

print(
    len(result[result["inbound"] == False])
)