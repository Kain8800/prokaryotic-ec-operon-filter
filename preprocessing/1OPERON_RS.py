import os

import pandas as pd

BASE_DIR = os.environ.get("PROJECT_BASE_DIR", ".")

path = os.path.join(BASE_DIR, "conserved_operon.download.txt")

# =============================
# 1. Read the file
# =============================
df = pd.read_csv(
    path,
    sep="\t",
    header=0,
    dtype="string",
    low_memory=False
)

# =============================
# 2. Keep the needed columns
# coid = real operon ID (CO000...)
# org  = taxon_id
# op   = gene list
# =============================
df = (
    df
    .rename(columns={
        "coid": "operon_id",
        "org": "taxon_id",
        "op": "genes"
    })[
        ["taxon_id", "operon_id", "genes"]
    ]
)

# =============================
# 3. Explode genes
# =============================
df["gene_id"] = df["genes"].str.split(",")
df = df.explode("gene_id").drop(columns="genes")

# =============================
# 4. Minimal cleanup
# =============================
df["gene_id"] = df["gene_id"].str.strip()
df = df.dropna(subset=["gene_id", "operon_id", "taxon_id"])

# Strip _RS
#df["gene_id_norm"] = df["gene_id"].str.replace("_RS", "_", regex=False)

# Drop duplicates
df = df.drop_duplicates(subset=["operon_id", "gene_id", "taxon_id"])

# =============================
# 5. Sanity print
# =============================
print(df.head(10))
print("Shape:", df.shape)
print("Top operon_id:")
print(df.operon_id.value_counts().head())

# =============================
# 6. Compute operon sizes
# =============================
operon_sizes = df.groupby("operon_id")["gene_id"].nunique()
print("Total operon_id:", len(operon_sizes))

# Keep only >=2 genes
valid_operons = operon_sizes[operon_sizes >= 2].index
df = df[df["operon_id"].isin(valid_operons)]
print("Operons with >=2 genes:", len(valid_operons))

# =============================
# 7. Dump taxids
# =============================
taxids = df["taxon_id"].unique()

with open(os.path.join(BASE_DIR, "taxids.txt"), "w") as f:
    for t in taxids:
        f.write(str(t) + "\n")

# =============================
# 8. Sanity check
# =============================
print("Single-gene operons:",
      (df.groupby("operon_id")["gene_id"].nunique() == 1).sum())

unique_taxon_ids = df["taxon_id"].unique()
# uniqueee = df["operon_id"].unique()
# unique1 = df["gene_id"].unique()
# print(f"operons {len(uniqueee)}")
# print(f"genes {len(unique1)}")

# n_genes = df["gene_id"].nunique()
# print("Unique genes:", n_genes)
# n_taxa = df["taxon_id"].nunique()
# print("Unique taxa:", n_taxa)
n_operons = df[["operon_id", "taxon_id"]].drop_duplicates().shape[0]
print("Unique operons:", n_operons)
print(f"Number of unique taxonomy IDs: {len(unique_taxon_ids)}")

# =============================
# 9. Save
# =============================
df.to_parquet(os.path.join(BASE_DIR, "operon_genes_RS.parquet"), index=False)
print(df.head())
print("Shape:", df.shape)