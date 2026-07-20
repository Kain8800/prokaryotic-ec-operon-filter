"""
KEGG_PATHWAY_EC_DOWNLOAD.py
Fetches the "KEGG pathway <-> EC number" mapping via the KEGG REST API.
Requests the full list of EC-to-reference-pathway links (link/ec/pathway)
and reduces the result to two columns: Path and EC_Numbers.

For the ec<->pathway link KEGG returns the pathway with an "enzyme" prefix
(ec00010). The 5-digit map number is the same across variants (map/ko/ec/rn),
so the letter prefix is normalized to rn to match the pathid_ec.tsv used in
the project (e.g. rn00010).

Run on a machine with internet access.

Input:  none (data is pulled from KEGG REST)
Output: pathid_ec.tsv  (columns: Path, EC_Numbers)
"""
import os
import requests
import pandas as pd
from io import StringIO
import time

BASE_DIR = os.environ.get("PROJECT_BASE_DIR", ".")

URL = "https://rest.kegg.jp/link/ec/pathway"
OUTPUT_FILE = os.path.join(BASE_DIR, "preprocessing", "pathid_ec.tsv")

# A single GET returns all pairs. KEGG can be flaky — small retry.
text = None
for attempt in range(3):
    try:
        response = requests.get(URL, timeout=120)
        response.raise_for_status()
        text = response.text
        break
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(15)
if not text or not text.strip():
    raise RuntimeError("Failed to fetch link/ec/pathway from KEGG after 3 attempts")

df = pd.read_csv(StringIO(text), sep="\t", header=None, names=["c0", "c1"])

# KEGG column order in link/<target>/<source> is source then target,
# i.e. path is expected in c0 and ec in c1. Detect by content so as not to
# depend on the order.
if df["c0"].iloc[0].startswith("path:"):
    df = df.rename(columns={"c0": "Path", "c1": "EC_Numbers"})
else:
    df = df.rename(columns={"c0": "EC_Numbers", "c1": "Path"})

# Strip the 'path:' and 'ec:' prefixes.
df["Path"] = df["Path"].str.replace("path:", "", regex=False)
df["EC_Numbers"] = df["EC_Numbers"].str.replace("ec:", "", regex=False)

# Normalize the map variant to rn: the 5-digit number is the same for map/ko/ec/rn.
df["Path"] = df["Path"].str.replace(r"^[a-z]+(\d{5})$", r"rn\1", regex=True)

df = df.drop_duplicates(subset=["Path", "EC_Numbers"])
df = df.sort_values(["Path", "EC_Numbers"]).reset_index(drop=True)

df.to_csv(OUTPUT_FILE, sep="\t", index=False)

print("=== KEGG pathway <-> EC ===")
print(f"Pairs fetched:        {len(df)}")
print(f"Unique pathways:    {df['Path'].nunique()}")
print(f"Unique EC:       {df['EC_Numbers'].nunique()}")
print(f"Written to:           {OUTPUT_FILE}")
