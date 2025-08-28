import pandas as pd
import re
import os
from pathlib import Path

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", " URL ", text)  # replace URLs
    text = re.sub(r"[^a-z0-9\s]", "", text)          # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(in_path, out_path):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(out_path)
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(in_path)
    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in dataset")

    df["review_clean"] = df["text"].apply(clean_text)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved cleaned data to {out_path}")

if __name__ == "__main__":
    # Ask user for input file
    in_path = input("Enter the path to your reviews CSV file: ").strip()

    # Default output file
    filename = os.path.basename(in_path).replace(".csv", "_clean.csv")
    out_path = os.path.join("data/processed", filename)

    preprocess(in_path, out_path)