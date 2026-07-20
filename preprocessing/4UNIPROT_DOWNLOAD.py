import requests
import pandas as pd
from io import StringIO
import time
import os

BASE_DIR = os.environ.get("PROJECT_BASE_DIR", ".")

# =============================
# 1. Load operon (RS version!)
# =============================
operon_df = pd.read_parquet(os.path.join(BASE_DIR, "operon_genes_RS.parquet"))

taxids = operon_df["taxon_id"].astype(str).unique()

print("Total unique taxids:", len(taxids))

# =============================
# 2. UniProt API settings
# =============================
BASE_URL = "https://rest.uniprot.org/uniprotkb/stream"

# added xref_refseq
FIELDS = "accession,organism_id,gene_names,xref_refseq,ec,sequence"

batch_size = 15
output_file = os.path.join(BASE_DIR, "uniprot_with_refseq.tsv")
checkpoint_file = os.path.join(BASE_DIR, "checkpoint_uniprot.txt")

# =============================
# 3. Checkpoint
# =============================
start_index = 0

if os.path.exists(checkpoint_file):
    with open(checkpoint_file) as f:
        start_index = int(f.read().strip())
    print(f"Resuming from batch index {start_index}")
else:
    print("Starting from scratch")

file_exists = os.path.exists(output_file)

# =============================
# 4. Main loop
# =============================
for i in range(start_index, len(taxids), batch_size):

    batch = taxids[i:i+batch_size]

    query = "(" + " OR ".join([f"organism_id:{t}" for t in batch]) + ")"

    params = {
        "format": "tsv",
        "fields": FIELDS,
        "query": query
    }

    print(f"Downloading batch {i}-{i+len(batch)}")

    try:
        response = requests.get(BASE_URL, params=params, timeout=300)
        response.raise_for_status()

        if response.text.strip():

            df = pd.read_csv(StringIO(response.text), sep="\t")

            if not df.empty:

                df.to_csv(
                    output_file,
                    sep="\t",
                    index=False,
                    mode="a",
                    header=not file_exists
                )

                file_exists = True

        # --- save checkpoint ---
        with open(checkpoint_file, "w") as f:
            f.write(str(i + batch_size))

        time.sleep(1.5)

    except Exception as e:
        print(f"Error in batch {i}: {e}")
        print("Retrying in 15 seconds...")
        time.sleep(15)
        continue

print("Download finished.")
