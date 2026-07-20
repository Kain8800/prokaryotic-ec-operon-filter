#!/usr/bin/env python3
# 0PATHID_PREPROCESS.py
# Removes three overview KEGG maps (01100 / 01110 / 01120) from pathid_ec.tsv:
# they link almost any EC and add noise to the operon filter. Idempotent —
# the original is backed up to pathid_ec.raw.tsv, reruns read from the backup.

import os
import re
import shutil
import pandas as pd


# =============================
# Paths
# =============================
BASE_DIR    = os.path.join(os.environ.get("PROJECT_BASE_DIR", "."), "preprocessing")
PATHID_PATH = os.path.join(BASE_DIR, "pathid_ec.tsv")
BACKUP_PATH = os.path.join(BASE_DIR, "pathid_ec.raw.tsv")


# Remove overview / super-aggregate KEGG maps:
# 01100 - Metabolic pathways
# 01110 - Biosynthesis of secondary metabolites
# 01120 - Microbial metabolism in diverse environments
OVERVIEW_RE = re.compile(
    r"^(?:rn|map|ko|path)?(?:01100|01110|01120)$",
    re.IGNORECASE
)


def main():
    # Read from the backup if it exists (idempotency).
    if os.path.isfile(BACKUP_PATH):
        source_path = BACKUP_PATH
        print(f"  [INFO] Reading original from backup: {BACKUP_PATH}")
    elif os.path.isfile(PATHID_PATH):
        source_path = PATHID_PATH
    else:
        raise FileNotFoundError(
            f"Found neither {PATHID_PATH} nor {BACKUP_PATH}"
        )

    df = pd.read_csv(source_path, sep="\t")
    n_in = len(df)

    # Find the pathway-ID column.
    if "Path" in df.columns:
        path_col = "Path"
    else:
        candidates = [c for c in df.columns
                      if c.lower() in ("path", "pathway", "path_id", "pathid", "kegg_id")]
        if not candidates:
            raise KeyError(
                f"KEGG path ID column not found. Columns in file: "
                f"{list(df.columns)}. Expected 'Path' or similar."
            )
        path_col = candidates[0]
        print(f"  [INFO] Using column '{path_col}' (not 'Path').")

    # Diagnostics: what will be removed.
    df[path_col] = df[path_col].astype(str).str.strip()
    mask_overview = df[path_col].apply(lambda x: bool(OVERVIEW_RE.match(x)))
    to_remove = df.loc[mask_overview, path_col].value_counts().sort_index()

    print("\n  IDs flagged as rn01100 (to be removed):")
    if len(to_remove) == 0:
        print("    (rn01100 not found in file)")
    else:
        for path_id, cnt in to_remove.items():
            print(f"    {path_id:>15}: {cnt} rows")

    # Filtering.
    df_clean = df.loc[~mask_overview].copy()
    n_out      = len(df_clean)
    n_removed  = n_in - n_out
    n_paths_in  = df[path_col].nunique()
    n_paths_out = df_clean[path_col].nunique()

    # Backup (only if not present yet).
    if not os.path.isfile(BACKUP_PATH):
        shutil.copy2(PATHID_PATH, BACKUP_PATH)
        print(f"\n  Backup created: {BACKUP_PATH}")
    else:
        print(f"\n  Backup already exists, not overwriting: {BACKUP_PATH}")

    df_clean.to_csv(PATHID_PATH, sep="\t", index=False)

    # Summary.
    print(f"\n=== pathid_ec.tsv preprocessing ===")
    print(f"Input rows:          {n_in}")
    print(f"Output rows:        {n_out} ({100*n_out/n_in:.1f}% of input)")
    print(f"Removed rows:        {n_removed}")
    print(f"Unique Path before:   {n_paths_in}")
    print(f"Unique Path after: {n_paths_out} ({n_paths_in - n_paths_out} removed)")
    print(f"Written to:           {PATHID_PATH}")


if __name__ == "__main__":
    main()