import os

import pandas as pd

BASE_DIR = os.environ.get("PROJECT_BASE_DIR", ".")

print("Loading UniProt...")

df = pd.read_csv(os.path.join(BASE_DIR, "uniprot_with_refseq.tsv"), sep="\t")

print("Initial shape:", df.shape)

# =============================
# 1. Drop rows without RefSeq
# =============================
df = df[df["RefSeq"].notna()]
df = df[df["RefSeq"] != ""]

print("After removing empty RefSeq:", df.shape)

# =============================
# 2. Split multiple RefSeq
# =============================
df["RefSeq"] = (
    df["RefSeq"]
    .astype(str)
    .str.replace(";", " ")
    .str.split()
)

df = df.explode("RefSeq")

df["RefSeq"] = df["RefSeq"].str.strip()

print("After explode:", df.shape)

# =============================
# 3. Keep only WP_*
# =============================
df = df[df["RefSeq"].str.startswith("WP_")]

print("After keeping WP_ only:", df.shape)

# =============================
# 4. Normalize EC
# =============================
df["EC number"] = (
    df["EC number"]
    .fillna("")
    .astype(str)
    .str.replace(";", " ")
    .str.split()
)

# multiple EC values are allowed,
# but drop fully empty ones
df = df[df["EC number"].str.len() > 0]

print("After EC cleaning:", df.shape)

# =============================
# 5. Drop duplicates
# =============================
df = df.drop_duplicates(
    subset=["Entry", "RefSeq"]
)

print("After deduplication:", df.shape)

# =============================
# 6. Save
# =============================
df.to_parquet(
    os.path.join(BASE_DIR, "uniprot_with_refseq_clean.parquet"),
    index=False
)

print("Saved cleaned UniProt file.")
