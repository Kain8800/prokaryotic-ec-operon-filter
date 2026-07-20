import os

import pandas as pd
import numpy as np
from collections import defaultdict

BASE_DIR = os.environ.get("PROJECT_BASE_DIR", ".")

print("Loading files...")

# 1. Operon
operon = pd.read_parquet(os.path.join(BASE_DIR, "operon_genes_RS.parquet"))
operon["taxon_id"] = operon["taxon_id"].astype(str)

# 2. locus → RefSeq
refseq_map = pd.read_parquet(os.path.join(BASE_DIR, "refseq_locus_to_protein.parquet"))
refseq_map["taxid"] = refseq_map["taxid"].astype(str)

refseq_map = refseq_map.rename(columns={
    "taxid": "taxon_id",
    "locus_tag": "gene_id"
})

# 3. UniProt cleaned
uniprot = pd.read_parquet(os.path.join(BASE_DIR, "uniprot_with_refseq_clean.parquet"))

print("Operon:", operon.shape)
print("RefSeq map:", refseq_map.shape)
print("UniProt:", uniprot.shape)
6372
# =============================
# 1. operon + locus→RefSeq
# =============================
print("\nMerging operon with RefSeq mapping...")

operon_refseq = operon.merge(
    refseq_map,
    on=["taxon_id", "gene_id"],
    how="inner"
)

print("After operon→RefSeq:", operon_refseq.shape)

# =============================
# 2. RefSeq → UniProt
# =============================
print("\nMerging with UniProt...")

merged = operon_refseq.merge(
    uniprot,
    left_on="refseq_protein_id",
    right_on="RefSeq",
    how="inner"
)

print("After RefSeq→UniProt:", merged.shape)

# =============================
# 3. Select the needed columns
# =============================
merged = merged[
    ["Entry", "taxon_id", "operon_id",
     "refseq_protein_id", "EC number", "Sequence"]
]

# drop duplicate Entry within one operon
merged = merged.drop_duplicates(
    subset=["Entry", "operon_id"]
)

print("After dedup:", merged.shape)

# =============================
# 4. Filter operons with >=2 proteins
# =============================
print("\nFiltering single-gene operons...")

operon_counts = defaultdict(set)

for op, entry in zip(merged["operon_id"], merged["Entry"]):
    operon_counts[op].add(entry)

valid_operons = {
    op for op, entries in operon_counts.items()
    if len(entries) >= 2
}

merged_filtered = merged[
    merged["operon_id"].isin(valid_operons)
]

print("Before filter:", merged.shape)
print("After filter:", merged_filtered.shape)

# =============================
# 5. Save
# =============================
merged_filtered.to_parquet(
    os.path.join(BASE_DIR, "merged_operon_uniprot_refseq_filtered.parquet"),
    index=False
)

print("\nSaved merged_operon_uniprot_refseq_filtered.parquet")



print(merged_filtered.head())
print(f"operon->RefSeq coverage: {len(operon_refseq)/len(operon)*100:.1f}%")
print(f"RefSeq->UniProt coverage: {len(merged)/len(operon_refseq)*100:.1f}%")
print(f"Total coverage from source operons: {len(merged_filtered)/len(operon)*100:.1f}%")

print(f"\nUnique proteins (Entry): {merged_filtered['Entry'].nunique()}")
print(f"Unique operons: {merged_filtered['operon_id'].nunique()}")
print(f"Unique organisms: {merged_filtered['taxon_id'].nunique()}")
print(f"Unique EC numbers: {merged_filtered['EC number'].explode().nunique()}")

op_sizes = merged_filtered.groupby("operon_id")["Entry"].nunique()
print(f"\nOperon sizes (proteins per operon):")
print(f"  min: {op_sizes.min()}, max: {op_sizes.max()}, median: {op_sizes.median():.0f}, mean: {op_sizes.mean():.1f}")
print(op_sizes.value_counts().sort_index().head(10))

seq_lens = merged_filtered["Sequence"].str.len()
print(f"\nSequence lengths (aa):")
print(f"  min: {seq_lens.min()}, max: {seq_lens.max()}, median: {seq_lens.median():.0f}, mean: {seq_lens.mean():.0f}")

ec_counts = merged_filtered["EC number"].apply(lambda x: len(x) if isinstance(x, list) else 1)
print(f"\nProteins with 1 EC: {(ec_counts == 1).sum()}")
print(f"Proteins with >=2 EC: {(ec_counts >= 2).sum()} ({(ec_counts >= 2).mean()*100:.1f}%)")

# Count proteins per operon BEFORE filtering
op_protein_counts = merged.groupby("operon_id")["Entry"].nunique()

print(f"\nOperons with >=2 proteins: {(op_protein_counts >= 2).sum()}")
print(f"Operons with <=1 protein: {(op_protein_counts <= 1).sum()}")
print(f"Proteins in operons with <=1 protein (loss): {merged[~merged['operon_id'].isin(valid_operons)].shape[0]}")

print(merged_filtered["EC number"].head(10))
print(type(merged_filtered["EC number"].iloc[0]))

ec_multi = uniprot["EC number"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 1)
print(f"In UniProt before merge - proteins with >=2 EC: {(ec_multi >= 2).sum()}")



print(merged_filtered.head(10))



multi_ec = uniprot[uniprot["EC number"].apply(len) >= 2]
matched = multi_ec[multi_ec["RefSeq"].isin(operon_refseq["refseq_protein_id"])]
print(f"Multi-EC in UniProt: {len(multi_ec)}")
print(f"Of them matched at merge: {len(matched)}")

raw = pd.read_csv(os.path.join(BASE_DIR, "uniprot_with_refseq.tsv"), sep="\t", nrows=1000)
multi = raw[raw["EC number"].str.contains(",", na=False)]
print(multi["EC number"].head(20))
