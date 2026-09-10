import pandas as pd

INPUT = "data/processed/apple_support_all.csv"
OUTPUT = "data/processed/apple_conversations.csv"

print("Loading AppleSupport data...")

df = pd.read_csv(INPUT)

# Make sure IDs are numeric
df["tweet_id"] = pd.to_numeric(
    df["tweet_id"],
    errors="coerce"
)

df["in_response_to_tweet_id"] = pd.to_numeric(
    df["in_response_to_tweet_id"],
    errors="coerce"
)

df = df.dropna(subset=["tweet_id"])

# Create lookup:
# tweet_id -> row
tweet_lookup = df.set_index("tweet_id").to_dict("index")

print(f"Tweets loaded: {len(df):,}")

# ---------------------------------------------------------
# Find conversation roots
# ---------------------------------------------------------

roots = []

for _, row in df.iterrows():

    parent = row["in_response_to_tweet_id"]

    # No parent = possible conversation root
    if pd.isna(parent):
        roots.append(row["tweet_id"])

    # Parent exists but isn't in our extracted data
    elif parent not in tweet_lookup:
        roots.append(row["tweet_id"])


print(f"Potential conversation roots: {len(roots):,}")


# ---------------------------------------------------------
# Build conversation chains
# ---------------------------------------------------------

conversations = []

for root_id in roots:

    current_id = root_id
    messages = []
    visited = set()

    while current_id in tweet_lookup:

        # Prevent accidental cycles
        if current_id in visited:
            break

        visited.add(current_id)

        row = tweet_lookup[current_id]

        messages.append({
            "tweet_id": current_id,
            "author_id": row["author_id"],
            "inbound": row["inbound"],
            "created_at": row["created_at"],
            "text": row["text"],
        })

        # Find next tweet in chain
        responses = row.get("response_tweet_id")

        if pd.isna(responses):
            break

        response_ids = str(responses).split(",")

        # Find the first response that exists
        next_id = None

        for response_id in response_ids:

            try:
                candidate = int(response_id.strip())

                if candidate in tweet_lookup:
                    next_id = candidate
                    break

            except ValueError:
                continue

        if next_id is None:
            break

        current_id = next_id

    # Keep meaningful conversations
    if len(messages) >= 2:

        conversations.append({
            "conversation_id": root_id,
            "message_count": len(messages),
            "messages": messages
        })


# ---------------------------------------------------------
# Convert conversations into rows
# ---------------------------------------------------------

output_rows = []

for conversation in conversations:

    conversation_id = conversation["conversation_id"]

    for position, message in enumerate(
        conversation["messages"]
    ):

        output_rows.append({
            "conversation_id": conversation_id,
            "position": position,
            "tweet_id": message["tweet_id"],
            "author_id": message["author_id"],
            "inbound": message["inbound"],
            "created_at": message["created_at"],
            "text": message["text"],
        })


result = pd.DataFrame(output_rows)

result.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 60)
print("CONVERSATION RECONSTRUCTION COMPLETE")
print("=" * 60)

print(f"Output: {OUTPUT}")
print(f"Conversation chains: {len(conversations):,}")
print(f"Conversation messages: {len(result):,}")

if len(conversations) > 0:

    lengths = [
        c["message_count"]
        for c in conversations
    ]

    print()
    print("Conversation length statistics:")
    print(f"Minimum: {min(lengths)}")
    print(f"Maximum: {max(lengths)}")
    print(f"Average: {sum(lengths) / len(lengths):.2f}")

print()
print("Inbound distribution:")

print(
    result["inbound"]
    .value_counts()
)

print()
print("Sample reconstructed conversation:")

if len(conversations) > 0:

    sample = conversations[0]

    print()
    print(
        f"Conversation ID: "
        f"{sample['conversation_id']}"
    )

    for message in sample["messages"]:

        speaker = (
            "CUSTOMER"
            if message["inbound"]
            else "APPLESUPPORT"
        )

        print()
        print(f"[{speaker}]")
        print(message["text"])