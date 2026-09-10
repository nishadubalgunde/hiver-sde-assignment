import re
import pandas as pd

INPUT_PATH = "data/processed/apple_customer_messages.csv"
OUTPUT_PATH = "data/processed/apple_silver_labeled.csv"
GOLDEN_PATH = "data/golden/apple_golden_set_200.csv"


def classify_message(text):
    text = str(text).lower().strip()

    # 1. iOS update
    if re.search(
        r"\b(ios|ipados)\b.*\b(update|upgrade)\b|\b(update|upgrade)\b.*\b(ios|ipados)\b"
        r"|\bsoftware update\b|\bcan't update\b|\bcannot update\b|\bfailed to update\b",
        text
    ):
        return "ios_update"

    # 2. Battery / power
    if re.search(
        r"\bbattery\b|\bcharging\b|\bcharger\b|\boverheat\w*\b|\boverheating\b"
        r"|\bpower\b|\bshut(?:s|ting)? down\b|\bwon't turn on\b|\bnot charging\b",
        text
    ):
        return "battery_power"

    # 3. Specific app issue
    if re.search(
        r"\b(app|application)\b.*\b(crash|crashing|freeze|freezing|won't open|"
        r"not opening|doesn't work|not working|fail|failed|loading)\b"
        r"|\b(crash|crashing|freeze|freezing)\b.*\b(app|application)\b",
        text
    ):
        return "app_issue"

    # 4. Audio / calls
    if re.search(
        r"\b(call|calling|phone call|facetime)\b|\b(microphone|mic|speaker|"
        r"volume|sound|audio|static|crackling)\b",
        text
    ):
        return "audio_call_issue"

    # 5. Connectivity
    if re.search(
        r"\b(wifi|wi-fi|bluetooth|airdrop|wireless|hotspot)\b"
        r"|\bwon't connect\b|\bcan't connect\b|\bconnection\b",
        text
    ):
        return "connectivity"

    # 6. Apple ID / account
    if re.search(
        r"\bapple id\b|\biclo[u?]d\b|\baccount\b.*\b(login|log in|sign in|"
        r"password|locked|access)\b|\bsign in\b|\blog in\b",
        text
    ):
        return "apple_id_account"

    # 7. App Store / purchases
    if re.search(
        r"\bapp store\b|\bpurchase\b|\brefund\b|\bcharged\b|\bpayment\b"
        r"|\bsubscription\b|\bdownload\b.*\bapp\b|\bcan't download\b",
        text
    ):
        return "app_store_purchase"

    # 8. Music / media
    if re.search(
        r"\bapple music\b|\bitunes\b|\bpodcast\b|\bbooks\b|\bibooks\b"
        r"|\bmusic\b.*\b(play|playing|sync|library)\b",
        text
    ):
        return "music_media"

    # 9. How-to / feature
    if re.search(
        r"\bhow do i\b|\bhow can i\b|\bhow to\b|\bwhere can i\b|\bwhere do i\b"
        r"|\bhow\b.*\b(turn on|turn off|enable|disable|change|set up)\b",
        text
    ):
        return "feature_how_to"

    # 10. Service / human support
    if re.search(
        r"\b(support|customer service|representative|agent|human)\b"
        r"|\brepair\b|\bappointment\b|\bcallback\b|\bcall me\b"
        r"|\bdm me\b|\bdirect message\b|\bcontact\b",
        text
    ):
        return "service_support"

    # 11. General device performance
    if re.search(
        r"\bslow\b|\blag\b|\blagging\b|\bfreez\w*\b|\bglitch\w*\b"
        r"|\bunresponsive\b|\bnot responding\b|\bcrash\w*\b",
        text
    ):
        return "device_performance"

    # 12. Everything unclear / unmatched
    return "other_unclear"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Load Golden Set IDs so they NEVER become training examples
    golden = pd.read_csv(GOLDEN_PATH)
    golden_ids = set(golden["tweet_id"].astype(str))

    df["tweet_id"] = df["tweet_id"].astype(str)

    # Remove Golden Set examples
    train_df = df[~df["tweet_id"].isin(golden_ids)].copy()

    # Apply weak/deterministic labeling rules
    train_df["intent"] = train_df["text"].apply(classify_message)

    # Keep useful columns
    output = train_df[
        ["tweet_id", "conversation_id", "created_at", "text", "intent"]
    ].copy()

    output.to_csv(OUTPUT_PATH, index=False)

    print("=== Silver Label Dataset ===")
    print(f"Original customer messages: {len(df)}")
    print(f"Golden examples excluded: {len(golden_ids)}")
    print(f"Silver training examples: {len(output)}")

    print("\n=== Intent Distribution ===")
    print(output["intent"].value_counts())

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()