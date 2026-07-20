import os
import pandas as pd
import gzip
import requests
from io import BytesIO

BASE_DIR = os.environ.get("PROJECT_BASE_DIR", ".")

# =============================
# 1. Load operon
# =============================
operon = pd.read_parquet(os.path.join(BASE_DIR, "operon_genes_RS.parquet"))
operon["taxon_id"] = operon["taxon_id"].astype(int)

# =============================
# 2. Load assemblies
# =============================
assemblies = pd.read_parquet(os.path.join(BASE_DIR, "selected_assemblies_with_gff.parquet"))

results = []

for _, row in assemblies.iterrows():

    taxid = row["taxid"]
    url = row["gff_url"]

    print(f"Processing taxid {taxid}")

    locus_set = set(
        operon[operon["taxon_id"] == taxid]["gene_id"]
    )

    if len(locus_set) == 0:
        continue

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with gzip.open(BytesIO(response.content), "rt") as f:

            for line in f:
                if line.startswith("#"):
                    continue

                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue

                if parts[2] != "CDS":
                    continue

                attrs = parts[8]

                if "locus_tag=" not in attrs:
                    continue

                attr_dict = dict(
                    field.split("=", 1)
                    for field in attrs.split(";")
                    if "=" in field
                )

                locus = attr_dict.get("locus_tag")
                protein = attr_dict.get("protein_id")

                if locus in locus_set:
                    results.append((taxid, locus, protein))

    except Exception as e:
        print(f"Failed {taxid}: {e}")

# =============================
# 3. Save
# =============================
mapping_df = pd.DataFrame(
    results,
    columns=["taxid", "locus_tag", "refseq_protein_id"]
)

mapping_df.to_parquet(os.path.join(BASE_DIR, "refseq_locus_to_protein.parquet"), index=False)

print("Done. Rows:", len(mapping_df))
