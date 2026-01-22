import pandas as pd
import re
import os

# ==========================================
# 1) Configure file path
# ==========================================
file_path = "/Users/xujiahe/Downloads/人类标注者答案.xlsx"

# ==========================================
# 2) Emotion lexicon (proxy list inspired by core NRC categories)
#    Includes proxies for Joy/Humor, Fear/Risk, Anger/Disgust, Sadness, etc.
# ==========================================
nrc_proxy = set([
    # --- Fear / Risk (often triggers verification) ---
    "risk", "risks", "risky", "danger", "dangerous", "threat", "panic", "fear", "afraid",
    "scared", "terror", "horror", "worry", "worried", "anxious", "warn", "warning",
    "harm", "harmful", "hurt", "kill", "death", "dead", "violence", "blood", "attack",
    "safe", "safety", "crisis", "emergency", "victim",

    # --- Anger / Disgust ---
    "hate", "angry", "mad", "rage", "disgust", "gross", "sick", "offensive", "insult",

    # --- Sadness ---
    "sad", "sorrow", "grief", "cry", "tragedy", "tragic", "suffering", "pity", "shame",

    # --- Joy / Humor (often associated with deprioritization) ---
    "happy", "joy", "fun", "funny", "laugh", "joke", "jokes", "prank", "humor", "sarcasm",
    "smile", "celebrate", "beautiful", "wonderful", "love", "like", "enjoy",

    # --- Surprise / Trust ---
    "surprise", "shock", "shocking", "amaze", "amazing", "trust", "believe", "faith", "truth"
])


def process_emotion_analysis(file_path: str) -> None:
    # --- A) Load data ---
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return

    try:
        # sheet_name=None reads all sheets
        sheets_dict = pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        print(f"ERROR: Failed to read Excel file: {e}")
        return

    all_dfs = []
    for sheet_name, df in sheets_dict.items():
        all_dfs.append(df)

    if not all_dfs:
        print("ERROR: No sheets were loaded (empty workbook).")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Successfully loaded data. Total rows: {len(full_df)}")

    # --- B) Basic cleaning ---
    # 1) Clean label (Q5)
    full_df["Q5"] = full_df["Q5"].astype(str).str.strip()

    def to_binary_label(x):
        lx = str(x).lower()
        if "not checkworthy" in lx:
            return 0
        if "checkworthy" in lx:
            return 1
        return None

    full_df["binary_label"] = full_df["Q5"].apply(to_binary_label)

    # Drop rows without valid labels
    full_df = full_df.dropna(subset=["binary_label"])

    # 2) Text preprocessing (merge Q6 + Q7)
    full_df["text"] = full_df["Q6"].fillna("") + " " + full_df["Q7"].fillna("")

    # Lowercase + remove punctuation
    full_df["text_clean"] = full_df["text"].str.lower().apply(
        lambda x: re.sub(r"[^\w\s]", " ", x)
    )

    # --- C) Emotion word matching ---
    def has_emotion_word(text: str) -> int:
        words = text.split()
        return 1 if any(w in nrc_proxy for w in words) else 0

    full_df["EmotionMention"] = full_df["text_clean"].apply(has_emotion_word)

    # --- D) Aggregate stats and print ---
    # Group: EmotionMention (0/1) vs checkworthy ratio
    result = full_df.groupby("EmotionMention")["binary_label"].agg(["count", "sum"])
    result["Checkworthy Ratio"] = result["sum"] / result["count"]

    result.columns = ["Total Samples", "Checkworthy Count", "Checkworthy Ratio"]
    result.index = ["No Emotion Words (0)", "Has Emotion Words (1)"]

    print("\n" + "=" * 40)
    print("Replication Output: Emotion Words vs Check-worthiness")
    print("=" * 40)
    print(result)
    print("=" * 40)


if __name__ == "__main__":
    process_emotion_analysis(file_path)
