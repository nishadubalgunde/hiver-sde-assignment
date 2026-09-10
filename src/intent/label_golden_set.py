import pandas as pd
import os

INPUT = "data/golden/apple_golden_set_200.csv"
OUTPUT = "data/golden/apple_golden_set_200.csv"

INTENTS = [
    "ios_update",
    "battery_power",
    "app_issue",
    "device_performance",
    "audio_call_issue",
    "connectivity",
    "apple_id_account",
    "app_store_purchase",
    "music_media",
    "feature_how_to",
    "service_support",
    "other_unclear",
]

df = pd.read_csv(INPUT)
df["intent"] = df["intent"].fillna("").astype(str)
df["label_notes"] = df["label_notes"].fillna("").astype(str)

# Resume from the first unlabeled example
start_index = 0

for i in range(len(df)):
    if pd.isna(df.loc[i, "intent"]) or str(df.loc[i, "intent"]).strip() == "":
        start_index = i
        break
else:
    print("All 200 examples are already labeled.")
    exit()

print("=" * 70)
print("APPLE SUPPORT GOLDEN SET — MANUAL LABELING")
print("=" * 70)
print()
print("Enter the number corresponding to the correct intent.")
print("Enter 'q' to save progress and exit.")
print()

for i in range(start_index, len(df)):

    row = df.iloc[i]

    print()
    print("=" * 70)
    print(f"EXAMPLE {i + 1} / {len(df)}")
    print("=" * 70)

    print()
    print("CONVERSATION CONTEXT:")
    print("-" * 70)
    print(row["conversation_context"])

    print()
    print("TARGET CUSTOMER MESSAGE:")
    print("-" * 70)
    print(row["target_message"])

    print()
    print("INTENTS:")
    for number, intent in enumerate(INTENTS, start=1):
        print(f"{number:2}. {intent}")

    while True:

        choice = input("\nYour choice: ").strip()

        if choice.lower() == "q":
            df.to_csv(OUTPUT, index=False)
            print()
            print("Progress saved.")
            print(f"Labeled: {i} / {len(df)}")
            exit()

        if choice.isdigit():
            number = int(choice)

            if 1 <= number <= len(INTENTS):
                selected_intent = INTENTS[number - 1]

                df.loc[i, "intent"] = selected_intent

                notes = input(
                    "Optional labeling note (press Enter to skip): "
                ).strip()

                df.loc[i, "label_notes"] = notes

                df.to_csv(OUTPUT, index=False)

                print()
                print(f"Saved: {selected_intent}")
                break

        print("Invalid choice. Enter a number from 1-12 or 'q'.")

print()
print("=" * 70)
print("GOLDEN SET LABELING COMPLETE")
print("=" * 70)

print()
print("Intent distribution:")
print(df["intent"].value_counts())

print()
print(f"Saved to: {OUTPUT}")