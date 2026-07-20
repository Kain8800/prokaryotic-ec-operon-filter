import os

import pandas as pd

BASE_DIR = os.environ.get("PROJECT_BASE_DIR", ".")

# =============================
# 1. Load assembly summary
# =============================

def load_assembly_summary(path):

    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=1,
        low_memory=False
    )

    df.columns = df.columns.str.replace("#", "", regex=False)
    df["taxid"] = df["taxid"].astype(int)

    return df


bacteria = load_assembly_summary(os.path.join(BASE_DIR, "assembly_summary_bacteria.txt"))
archaea  = load_assembly_summary(os.path.join(BASE_DIR, "assembly_summary_archaea.txt"))

assembly_df = pd.concat([bacteria, archaea], ignore_index=True)


# =============================
# 2. Load taxids from the operon dataset
# =============================

taxids = set(
    pd.read_csv(os.path.join(BASE_DIR, "taxids.txt"), header=None, dtype=int)[0]
)


# =============================
# 3. Keep only genomes used in operons
# =============================

assembly_df = assembly_df[assembly_df["taxid"].isin(taxids)]


# =============================
# 4. Remove metagenomes
# =============================

assembly_df = assembly_df[
    (assembly_df["excluded_from_refseq"].isna()) |
    (assembly_df["excluded_from_refseq"] == "na")
]


# =============================
# 5. Select the best assembly per taxid
# =============================

def select_best_assembly(df):

    ref = df[df["refseq_category"] == "reference genome"]
    if len(ref):
        return ref.iloc[0]

    rep = df[df["refseq_category"] == "representative genome"]
    if len(rep):
        return rep.iloc[0]

    complete = df[df["assembly_level"] == "Complete Genome"]
    if len(complete):
        return complete.iloc[0]

    latest = df[df["version_status"] == "latest"]
    if len(latest):
        return latest.iloc[0]

    return df.iloc[0]


selected = (
    assembly_df
    .groupby("taxid", group_keys=False)
    .apply(select_best_assembly)
    .reset_index(drop=True)
)


# =============================
# 6. Build GFF URLs
# =============================

def build_gff_url(row):

    base = row["ftp_path"].split("/")[-1]
    return f"{row['ftp_path']}/{base}_genomic.gff.gz"


selected["gff_url"] = selected.apply(build_gff_url, axis=1)


# =============================
# 7. Save
# =============================

selected[["taxid", "assembly_accession", "gff_url"]].to_parquet(
    os.path.join(BASE_DIR, "selected_assemblies_with_gff.parquet"),
    index=False
)

print("Saved selected_assemblies_with_gff.parquet")