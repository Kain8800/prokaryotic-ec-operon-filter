"""
eval_common.py

Shared code for EC-number prediction evaluation: building the evaluation set,
homology stratification (MMseqs2), parsing ProteInfer and CLEAN predictions,
the operon filter (OpeRONScorer) and metrics. Used by 7_EVALUATION.py.

Paths come from the environment variables PROJECT_BASE_DIR (data root; defaults
to the current directory) and MMSEQS_BIN (path to the MMseqs2 executable;
defaults to mmseqs from PATH).

RESULTS_DIR is not set here — it is set by the evaluation script
(7_EVALUATION.py) so the results directory sits next to its output.
"""
import os
import struct
import subprocess
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.environ.get("PROJECT_BASE_DIR", ".")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation2")

PARQUET_PATH = os.path.join(BASE_DIR,
    "merged_operon_uniprot_refseq_filtered.parquet")
PATHID_EC_PATH = os.path.join(BASE_DIR, "preprocessing", "pathid_ec.tsv")
PI_TFRECORD_PATH = os.path.join(BASE_DIR,
    "research_proteins_proteinfer_datasets_swissprot_clustered_train.tfrecord")
CLEAN_TRAIN_PATH = os.path.join(BASE_DIR, "split100.csv")
PI_PRED_PATH = os.path.join(BASE_DIR, "proteinfer_predictions2.tsv")
CLEAN_PRED_PATH = os.path.join(BASE_DIR, "query_maxsep.csv")

MMSEQS_BIN = os.environ.get("MMSEQS_BIN", "mmseqs")
MMSEQS_WORK_DIR = os.path.join(OUTPUT_DIR, "mmseqs_work")
EVAL_FASTA_PATH = os.path.join(OUTPUT_DIR, "eval_sequences.fasta")
TRAIN_FASTA_PATH = os.path.join(OUTPUT_DIR, "train_combined.fasta")

# Stratum boundaries by maximum identity to the training set;
# the strata follow the standard split into homology groups.
IDENTITY_THRESHOLDS = {"HIGH": 0.70, "MEDIUM": 0.40}
STRATA_ORDER = ["HIGH", "MEDIUM", "LOW", "NONE"]

# Sentinel value for proteins without operon context: their predictions
# pass the binary filter without a check.
NO_CONTEXT_SCORE = float("inf")
BINARY_TAU = 0.5
CONF_THRESHOLDS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]
CLEAN_PARSE_MIN = 1e-9


def normalize_ec(s):
    return s.strip().replace("EC:", "").replace("ec:", "").strip()


def is_full_ec(ec):
    p = ec.split(".")
    return len(p) == 4 and all(x.isdigit() for x in p)


def ec_at_level(ec, level):
    return ".".join(ec.split(".")[:level])


def parse_tfrecord_sequences(path):
    """Extract sequences from a ProteInfer TFRecord.

    First tries the TensorFlow API. If it is missing,
    it falls back to manual parsing of the TFRecord protobuf format to extract
    the 'sequence' field without depending on the heavy library.
    """
    print(f"\n  Loading train ProteInfer: {path}")
    sequences = set()
    try:
        import tensorflow as tf
        raw = tf.data.TFRecordDataset(path)
        spec = {"sequence": tf.io.FixedLenFeature([], tf.string)}
        cnt = 0
        for rec in raw:
            seq = tf.io.parse_single_example(rec, spec)["sequence"]
            seq = seq.numpy().decode("utf-8")
            sequences.add(seq); cnt += 1
            if cnt % 50000 == 0: print(f"      {cnt} records...")
        print(f"    {len(sequences)} unique of {cnt}")
        return sequences
    except ImportError:
        print("    Manual protobuf parsing...")

    # Manual parsing of the TFRecord protobuf format. Functions with short names
    # implement decoding primitives: read record (_rd), varint (_dv),
    # packed byte list (_pbl), a single field (_pf), a
    # MapEntry (_pme), a feature dict (_pfs) and Example (_pe).
    def _rd(f):
        b = f.read(8)
        if len(b) < 8: return None
        ln = struct.unpack("<Q", b)[0]; f.read(4)
        d = f.read(ln); f.read(4)
        return d

    def _dv(d, p):
        r = s = 0
        while p < len(d):
            b = d[p]; p += 1; r |= (b & 0x7F) << s
            if not (b & 0x80): break
            s += 7
        return r, p

    def _pbl(d):
        v, p = [], 0
        while p < len(d):
            t, p = _dv(d, p)
            if (t & 7) == 2:
                l, p = _dv(d, p); v.append(d[p:p+l]); p += l
            else:
                break
        return v

    def _pf(d):
        p = 0
        while p < len(d):
            t, p = _dv(d, p)
            if (t & 7) == 2:
                l, p = _dv(d, p); pl = d[p:p+l]; p += l
                if (t >> 3) == 1: return _pbl(pl)
            elif (t & 7) == 0:
                _, p = _dv(d, p)
            else:
                break

    def _pme(d):
        k = v = None; p = 0
        while p < len(d):
            t, p = _dv(d, p)
            if (t & 7) == 2:
                l, p = _dv(d, p); pl = d[p:p+l]; p += l
                if (t >> 3) == 1: k = pl.decode("utf-8")
                elif (t >> 3) == 2: v = _pf(pl)
            elif (t & 7) == 0:
                _, p = _dv(d, p)
            else:
                break
        return k, v

    def _pfs(d):
        fs, p = {}, 0
        while p < len(d):
            t, p = _dv(d, p)
            if (t & 7) == 2 and (t >> 3) == 1:
                l, p = _dv(d, p); k2, v2 = _pme(d[p:p+l]); p += l
                if k2: fs[k2] = v2
            elif (t & 7) == 2:
                l, p = _dv(d, p); p += l
            elif (t & 7) == 0:
                _, p = _dv(d, p)
            else:
                break
        return fs

    def _pe(d):
        fs, p = {}, 0
        while p < len(d):
            t, p = _dv(d, p)
            if (t & 7) == 2:
                l, p = _dv(d, p)
                if (t >> 3) == 1: fs = _pfs(d[p:p+l])
                p += l
            elif (t & 7) == 0:
                _, p = _dv(d, p)
            else:
                break
        return fs

    cnt = 0
    with open(path, "rb") as f:
        while True:
            rec = _rd(f)
            if rec is None: break
            try:
                fs = _pe(rec)
                if "sequence" in fs and fs["sequence"]:
                    sb = fs["sequence"]
                    if isinstance(sb, list) and sb:
                        sq = sb[0].decode("utf-8")
                    elif isinstance(sb, bytes):
                        sq = sb.decode("utf-8")
                    else:
                        sq = None
                    if sq: sequences.add(sq)
            except: pass
            cnt += 1
            if cnt % 50000 == 0: print(f"      {cnt} records...")
    print(f"    {len(sequences)} unique of {cnt}")
    return sequences


def load_clean_train(path):
    print(f"\n  Loading train CLEAN: {path}")
    df = pd.read_csv(path, sep="\t")
    col = next((c for c in ["Sequence", "sequence", "seq"]
                if c in df.columns), None)
    if not col: return set()
    seqs = set(df[col].dropna().astype(str).str.strip())
    print(f"    {len(seqs)} sequences")
    return seqs


def load_all_train():
    print(f"\n{'='*70}\nLoading train")
    pi = (parse_tfrecord_sequences(PI_TFRECORD_PATH)
          if os.path.isfile(PI_TFRECORD_PATH) else set())
    cl = (load_clean_train(CLEAN_TRAIN_PATH)
          if os.path.isfile(CLEAN_TRAIN_PATH) else set())
    # Union of ProteInfer and CLEAN training sets: a protein counts as training
    # for later exclusion from eval if it appears in at least one
    # of the two sets (strict per-sequence exclusion rule).
    t = pi | cl
    print(f"\n  PI: {len(pi)}, CLEAN: {len(cl)}, "
          f"intersection: {len(pi&cl)}, total: {len(t)}")
    return t


def _boundary_gene_dedup(df):
    """Boundary-gene deduplication: each protein (Entry) within an
    organism (taxon_id) is bound to exactly one operon.

    The largest operon is chosen; ties are broken
    stably by operon_id. Without this step, boundary genes belonging
    to several overlapping operons at once inflate the
    count of unique operons by their multiplicity.
    """
    op_key = ["operon_id", "taxon_id"]
    op_size = (df.drop_duplicates(subset=op_key + ["Entry"])
                 .groupby(op_key, observed=True)["Entry"]
                 .size()
                 .rename("op_size")
                 .reset_index())
    pairs = (df.drop_duplicates(subset=["Entry"] + op_key)
               .merge(op_size, on=op_key, how="left", validate="m:1"))
    pairs = pairs.sort_values(["op_size", "operon_id"],
                              ascending=[False, True])
    kept = pairs.drop_duplicates(subset=["Entry", "taxon_id"],
                                 keep="first")
    chosen = kept[["Entry", "operon_id", "taxon_id"]]
    return df.merge(chosen, on=["Entry", "operon_id", "taxon_id"],
                    how="inner")


def _drop_operon_singletons(df, op_key=("operon_id", "taxon_id"),
                            value_col="Entry"):
    """Remove operons that contain fewer than two unique values of
    value_col. Returns the filtered df and the number of removed single-gene operons.
    """
    op_key = list(op_key)
    sz = (df.drop_duplicates(subset=[value_col] + op_key)
            .groupby(op_key, observed=True)[value_col]
            .size())
    valid = sz[sz >= 2].index
    n_removed = int((sz == 1).sum())
    df = df.set_index(op_key)
    df = df[df.index.isin(valid)].reset_index()
    return df, n_removed


def _n_op(df):
    """Number of unique (operon_id, taxon_id) pairs - the operon identifier."""
    return df.drop_duplicates(subset=["operon_id", "taxon_id"]).shape[0]


def build_eval_dataset(train_sequences):
    """Staged construction of the evaluation set following the steps in section 2.5
    of the methods. Each stage prints the number of proteins and operons
    for later cross-checking with the thesis.
    """
    print(f"\n{'#'*70}\n# BUILDING EVAL DATASET "
          f"(boundary-gene dedup on input)\n{'#'*70}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"\n  Loaded: {len(df)} rows, "
          f"{df['Entry'].nunique()} Entry, "
          f"{_n_op(df)} operon_id+taxon_id pairs")

    print(f"\n  --- Boundary-gene dedup (input) ---")
    df = _boundary_gene_dedup(df)
    df, n_art = _drop_operon_singletons(df, value_col="Entry")
    print(f"  Singleton artifacts removed: {n_art}")
    print(f"  After dedup: {df['Entry'].nunique()} proteins, "
          f"{_n_op(df)} operons")

    # EC numbers are coerced to a tuple for a later explode.
    # The field may be a string (one EC), a list, a numpy array or NaN -
    # all variants are normalized to a tuple.
    df["EC number"] = df["EC number"].apply(
        lambda x: tuple(x) if isinstance(x, (np.ndarray, list, tuple))
        else (x,))

    # Strict exclusion: ALL proteins whose sequence
    # appears in at least one training set are removed. This guarantees
    # no leakage and lets the results be interpreted correctly.
    if train_sequences:
        m = df["Sequence"].isin(train_sequences)
        print(f"\n  Train: removed {m.sum()} rows")
        df = df[~m].copy()
        print(f"  After removing train: {df['Entry'].nunique()} proteins, "
              f"{_n_op(df)} operons")

    df = df.drop_duplicates(
        subset=["Sequence", "operon_id", "taxon_id"]).copy()
    print(f"  After drop_duplicates: {df['Entry'].nunique()} proteins, "
          f"{_n_op(df)} operons")

    df, n_sg = _drop_operon_singletons(df, value_col="Sequence")
    print(f"  Single-gene operons after removing train: {n_sg}")
    print(f"  After removing single-gene operons: {df['Entry'].nunique()} proteins, "
          f"{_n_op(df)} operons")

    df = df.explode("EC number").copy()
    df["EC number"] = df["EC number"].astype(str)
    df = df[(df["EC number"] != "nan")
            & (df["EC number"].str.strip() != "")].copy()

    # Enrich the evaluation set with KEGG metabolic pathway annotations.
    print(f"\n  --- Metabolic pathway enrichment ---")
    pathid = pd.read_csv(PATHID_EC_PATH, sep="\t")
    print(f"  File pathid_ec.tsv:")
    print(f"    Records (EC -> Path): {len(pathid)}")
    print(f"    Unique EC numbers: {pathid['EC_Numbers'].nunique()}")
    print(f"    Unique KEGG pathways: {pathid['Path'].nunique()}")

    ec_in_dataset = set(df["EC number"].unique())
    ec_in_pathid = set(pathid["EC_Numbers"].unique())
    ec_covered = ec_in_dataset & ec_in_pathid
    ec_not_covered = ec_in_dataset - ec_in_pathid
    print(f"  Dataset EC-number coverage:")
    print(f"    EC in dataset: {len(ec_in_dataset)}")
    print(f"    EC with paths: {len(ec_covered)} "
          f"({100*len(ec_covered)/len(ec_in_dataset):.1f}%)")
    print(f"    EC without paths: {len(ec_not_covered)} "
          f"({100*len(ec_not_covered)/len(ec_in_dataset):.1f}%)")

    df = df.merge(pathid, left_on="EC number",
                  right_on="EC_Numbers", how="left")
    if "EC_Numbers" in df.columns:
        df = df.drop(columns=["EC_Numbers"])

    n_with_path = df["Path"].notna().sum()
    n_without_path = df["Path"].isna().sum()
    print(f"  Record coverage:")
    print(f"    Records with paths: {n_with_path} "
          f"({100*n_with_path/len(df):.1f}%)")
    print(f"    Records without paths: {n_without_path} "
          f"({100*n_without_path/len(df):.1f}%)")

    # Remove uninformative operons: operons that, after filtering
    # by presence of a Path annotation, have fewer than two proteins. Such operons
    # provide no neighbor-gene context for the filter.
    print(f"\n  --- Removing uninformative operons ---")
    df_wp = df[df["Path"].notna()].copy()
    ppo = (df_wp.drop_duplicates(subset=["Entry", "operon_id", "taxon_id"])
                .groupby(["operon_id", "taxon_id"], observed=True)["Entry"]
                .nunique())
    valid = set(ppo[ppo >= 2].index)
    df_idx = df.set_index(["operon_id", "taxon_id"])
    all_ops = set(df_idx.index.unique())
    eff_sg = all_ops - valid

    n_before = _n_op(df)
    n_proteins_before = df["Entry"].nunique()
    df = df_idx[df_idx.index.isin(list(valid))].reset_index()
    n_after = _n_op(df)
    n_proteins_after = df["Entry"].nunique()

    print(f"  Before filtering:")
    print(f"    Operons: {n_before}")
    print(f"    Proteins: {n_proteins_before}")
    print(f" Uninformative operons "
          f"(< 2 proteins with paths): {len(eff_sg)}")
    print(f"  After filtering:")
    print(f"    Operons: {n_after} ({100*n_after/n_before:.1f}%)")
    print(f"    Proteins: {n_proteins_after} "
          f"({100*n_proteins_after/n_proteins_before:.1f}%)")

    # Final deduplication by the triple (Sequence, operon_id, Path).
    # Filling NaN with the __NO_PATH__ marker keeps rows without a path
    # from collapsing into duplicates by the first two fields.
    df["_pk"] = df["Path"].fillna("__NO_PATH__")
    df = df.drop_duplicates(
        subset=["Sequence", "operon_id", "_pk"]).drop(columns=["_pk"]).copy()
    print(f"  Result: {len(df)} rows, {df['Sequence'].nunique()} seqs, "
          f"{_n_op(df)} operons")

    # FASTA for MMseqs2: unique sequences keyed by the Entry identifier.
    uq = df.drop_duplicates(subset=["Sequence"])[["Entry", "Sequence"]]
    with open(EVAL_FASTA_PATH, "w") as f:
        for _, r in uq.iterrows():
            f.write(f">{r['Entry']}\n{r['Sequence']}\n")
    print(f"  FASTA: {len(uq)} seqs")
    return df


def save_train_fasta(train):
    with open(TRAIN_FASTA_PATH, "w") as f:
        for i, s in enumerate(train):
            f.write(f">train_{i}\n{s}\n")
    print(f"  Train FASTA: {len(train)} seqs")


def run_mmseqs2(eval_df):
    """Compare all eval sequences against all train sequences to determine
    the maximum identity and split eval into strata HIGH/MEDIUM/LOW/NONE.

    Uses a cache in MMSEQS_WORK_DIR: if result.tsv exists and is non-empty,
    the search is not rerun. If MMseqs2 is unavailable, all
    sequences are assigned to the ALL stratum.
    """
    print(f"\n{'#'*70}\n# STRATIFICATION (MMSeqs2)\n{'#'*70}")
    os.makedirs(MMSEQS_WORK_DIR, exist_ok=True)
    result_tsv = os.path.join(MMSEQS_WORK_DIR, "result.tsv")

    if os.path.isfile(result_tsv) and os.path.getsize(result_tsv) > 0:
        print(f"  [CACHE] result.tsv "
              f"({os.path.getsize(result_tsv)/1e6:.1f} MB)")
    else:
        if not os.path.isfile(MMSEQS_BIN):
            return _fallback_strata(eval_df)
        import shutil
        # Clean the working directory before a new search.
        for it in os.listdir(MMSEQS_WORK_DIR):
            p = os.path.join(MMSEQS_WORK_DIR, it)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)

        tdb = os.path.join(MMSEQS_WORK_DIR, "train_db")
        edb = os.path.join(MMSEQS_WORK_DIR, "eval_db")
        rdb = os.path.join(MMSEQS_WORK_DIR, "result")
        tmp = os.path.join(MMSEQS_WORK_DIR, "tmp")
        os.makedirs(tmp, exist_ok=True)

        def run_cmd(cmd, desc):
            print(f"  [{desc}]")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                print(f"  [ERR] {r.stderr[:300]}")
                return False
            return True

        # Search parameters: high sensitivity s=7.5 to find
        # distant homologs; e-value 10 (loose threshold) - the actual
        # stratum membership criterion is applied later via identity.
        commands = [
            ([MMSEQS_BIN, "createdb", TRAIN_FASTA_PATH, tdb],
             "createdb train"),
            ([MMSEQS_BIN, "createdb", EVAL_FASTA_PATH, edb],
             "createdb eval"),
            ([MMSEQS_BIN, "search", edb, tdb, rdb, tmp,
              "--min-seq-id", "0.0", "-s", "7.5",
              "--max-seqs", "300", "-e", "10", "--threads", "8"],
             "search"),
            ([MMSEQS_BIN, "convertalis", edb, tdb, rdb, result_tsv,
              "--format-output", "query,target,fident,alnlen,evalue"],
             "convertalis"),
        ]
        for cmd, desc in commands:
            if not run_cmd(cmd, desc):
                return _fallback_strata(eval_df)

    if not os.path.isfile(result_tsv) or os.path.getsize(result_tsv) == 0:
        return _fallback_strata(eval_df)

    hits = pd.read_csv(result_tsv, sep="\t", header=None,
                       names=["query", "target", "fident", "alnlen", "evalue"])
    print(f"  Hits: {len(hits)}, query: {hits['query'].nunique()}")

    # For each eval protein - the maximum identity to any train protein.
    info = hits.groupby("query")["fident"].max().reset_index()
    info.columns = ["Entry", "max_identity"]

    def assign_group(x):
        # NONE - no hit at all (even a loose one) to train.
        # HIGH/MEDIUM/LOW - by the boundaries set in IDENTITY_THRESHOLDS.
        if pd.isna(x): return "NONE"
        if x > IDENTITY_THRESHOLDS["HIGH"]: return "HIGH"
        if x >= IDENTITY_THRESHOLDS["MEDIUM"]: return "MEDIUM"
        return "LOW"

    info["identity_group"] = info["max_identity"].apply(assign_group)
    eval_df = eval_df.merge(info, on="Entry", how="left")

    # Proteins without hits are missing from info after merge; fill with defaults.
    for col, val in [("max_identity", 0.0),
                     ("identity_group", "NONE")]:
        eval_df[col] = eval_df[col].fillna(val)

    gc = eval_df.drop_duplicates("Entry").groupby("identity_group").size()
    tot = eval_df["Entry"].nunique()
    for g in STRATA_ORDER:
        print(f"  {g:>6}: {gc.get(g,0):>6} ({100*gc.get(g,0)/tot:.1f}%)")
    return eval_df


def _fallback_strata(df):
    print("  [FALLBACK] -> ALL")
    df["max_identity"] = np.nan
    df["identity_group"] = "ALL"
    return df


def parse_pi_predictions(path):
    """Parse ProteInfer predictions. From each line it extracts
    the protein identifier, the predicted EC number and the model confidence.
    Within a protein, predictions are sorted by descending confidence.
    """
    print(f"\n  Parsing ProteInfer: {path}")
    df = pd.read_csv(path, sep="\t")
    preds = defaultdict(list)
    for _, row in df.iterrows():
        ec = normalize_ec(str(row["predicted_label"]))
        if is_full_ec(ec):
            preds[str(row["sequence_name"]).strip()].append(
                (ec, float(row["confidence"])))
    for e in preds:
        preds[e].sort(key=lambda x: -x[1])
    print(f"    Proteins: {len(preds)}")
    return dict(preds)


def parse_clean_predictions(path, min_conf=CLEAN_PARSE_MIN):
    """Parse CLEAN predictions. Format - CSV: each line has a protein
    and a comma-separated list of EC/score predictions. Predictions with
    confidence below min_conf are dropped during parsing.
    """
    print(f"\n  Parsing CLEAN: {path} (min_conf={min_conf})")
    preds = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(",")
            entry = parts[0].strip()
            for part in parts[1:]:
                part = part.strip()
                if "/" not in part: continue
                ec_part, score_part = part.rsplit("/", 1)
                try:
                    score = float(score_part)
                except ValueError:
                    continue
                if score < min_conf: continue
                ec = normalize_ec(ec_part)
                if is_full_ec(ec):
                    preds[entry].append((ec, score))
    for e in preds:
        preds[e].sort(key=lambda x: -x[1])
    print(f"    Proteins: {len(preds)}")
    return dict(preds)


class PredictionCache:
    """Cache for threshold-filtered predictions.

    The same threshold appears in many sections of the script; to
    avoid recomputing the filtering each time, results for each
    threshold value are cached. Likewise for the filtered
    versions (pi_bin, cl_bin).
    """

    def __init__(self, pi_preds, clean_preds, all_entries, scorer):
        self.pi_raw = pi_preds
        self.cl_raw = clean_preds
        self.all_e = all_entries
        self.scorer = scorer
        self._pi_cache = {}
        self._cl_cache = {}
        self._pi_bin_cache = {}
        self._cl_bin_cache = {}

    def _filter(self, preds, thr):
        # Keep proteins whose most confident prediction
        # has confidence >= thr and which belong to the evaluation set.
        return {e: c for e, c in preds.items()
                if c and c[0][1] >= thr and e in self.all_e}

    def pi(self, thr):
        if thr not in self._pi_cache:
            self._pi_cache[thr] = self._filter(self.pi_raw, thr)
        return self._pi_cache[thr]

    def cl(self, thr):
        if thr not in self._cl_cache:
            self._cl_cache[thr] = self._filter(self.cl_raw, thr)
        return self._cl_cache[thr]

    def pi_bin(self, thr):
        if thr not in self._pi_bin_cache:
            self._pi_bin_cache[thr] = self.scorer.apply_binary_reject(
                self.pi(thr))
        return self._pi_bin_cache[thr]

    def cl_bin(self, thr):
        if thr not in self._cl_bin_cache:
            self._cl_bin_cache[thr] = self.scorer.apply_binary_reject(
                self.cl(thr))
        return self._cl_bin_cache[thr]


class OpeRONScorer:
    """Operon filter with adaptive behavior: proteins without
    operon context pass without a check.

    For each protein the filter checks two conditions (methods, section 2.9):
      1) the predicted EC number participates in at least one metabolic
         pathway shared with the EC numbers of neighboring operon proteins;
      2) or it matches the EC number of at least one neighbor.
    If either condition holds, the prediction is accepted.
    """

    def __init__(self, eval_df):
        print(f"\n{'='*70}\nOpeRONScorer (adaptive reject)")
        self.eval_df = eval_df
        self.ec_to_paths = self._build_ec_paths()
        self.operon_entry_paths = defaultdict(lambda: defaultdict(set))
        self.operon_entry_ecs = defaultdict(lambda: defaultdict(set))
        self.entry_to_operon = {}
        self._build()
        self._print_stats()

    def _build_ec_paths(self):
        # Mapping from an EC number to the set of KEGG pathways it participates in.
        r = defaultdict(set)
        for _, row in self.eval_df.iterrows():
            ec = normalize_ec(str(row["EC number"]))
            p = row.get("Path")
            if pd.notna(p) and str(p).strip():
                r[ec].add(str(p).strip())
        return dict(r)

    def _build(self):
        # Indexes for fast lookup of neighbors and their annotations by operon
        # identifier: pathways and EC numbers of each protein in the operon.
        for _, row in self.eval_df.iterrows():
            oid, entry = row["operon_id"], row["Entry"]
            self.entry_to_operon[entry] = oid
            p = row.get("Path")
            if pd.notna(p) and str(p).strip():
                self.operon_entry_paths[oid][entry].add(str(p).strip())
            ec = normalize_ec(str(row["EC number"]))
            if is_full_ec(ec):
                self.operon_entry_ecs[oid][entry].add(ec)
        print(f"  Operons: {len(self.operon_entry_paths)}, "
              f"proteins: {len(self.entry_to_operon)}")

    def _print_stats(self):
        n_ctx = n_no = 0
        for entry, oid in self.entry_to_operon.items():
            if self._neighbor_tf(entry, oid):
                n_ctx += 1
            else:
                n_no += 1
        tot = n_ctx + n_no
        print(f"  With context: {n_ctx} ({100*n_ctx/tot:.1f}%), "
              f"without: {n_no} ({100*n_no/tot:.1f}%)")
        print(f"  OpeRONScorer ready.\n")

    def _neighbor_tf(self, entry, oid):
        # Fractions of operon neighbors annotated with each pathway. Returns
        # a dict {path: fraction of neighbors in that pathway} or an empty dict,
        # if the operon has no annotated neighbors.
        ep = self.operon_entry_paths.get(oid, {})
        nb = {e: ps for e, ps in ep.items() if e != entry}
        n = len(nb)
        if n == 0:
            return {}
        pc = defaultdict(int)
        for e, ps in nb.items():
            for p in ps:
                pc[p] += 1
        return {p: c/n for p, c in pc.items()}

    def _neighbor_ecs(self, entry, oid):
        # Union of EC numbers of all operon neighbors.
        r = set()
        for e, ecs in self.operon_entry_ecs.get(oid, {}).items():
            if e != entry:
                r.update(ecs)
        return r

    def score_binary(self, ec, entry):
        # Returns 1.0 if the prediction agrees with the operon context
        # (shared pathway or EC match with a neighbor), 0.0 otherwise. For proteins
        # without context returns NO_CONTEXT_SCORE - bypassing the filter.
        oid = self.entry_to_operon.get(entry)
        if not oid:
            return NO_CONTEXT_SCORE
        ntf = self._neighbor_tf(entry, oid)
        if not ntf:
            return NO_CONTEXT_SCORE
        ecp = self.ec_to_paths.get(ec, set())
        if not ecp:
            return 0.0
        if ec in self._neighbor_ecs(entry, oid):
            return 1.0
        return 1.0 if ecp & set(ntf.keys()) else 0.0

    def apply_binary_reject(self, preds, tau=BINARY_TAU):
        # Apply the filter: keep predictions with score >= tau.
        # For predictions without context score_binary returns inf,
        # which is always >= tau.
        return {e: c for e, c in preds.items()
                if c and self.score_binary(c[0][0], e) >= tau}


def build_ground_truth(eval_df):
    # True EC numbers of each protein in the evaluation set.
    gt = defaultdict(set)
    for _, row in eval_df.iterrows():
        ec = normalize_ec(str(row["EC number"]))
        if is_full_ec(ec):
            gt[row["Entry"]].add(ec)
    return dict(gt)


def metrics(preds, gt, entries, lvl=4):
    """Compute prediction-quality metrics at a given EC hierarchy level.

    Returns accuracy, F1 macro and coverage.
    """
    yt, yp = [], []
    nt, np_, nc = len(entries), 0, 0
    for e in entries:
        g = gt.get(e, set())
        if not g: continue
        # True EC truncated to the given hierarchy level.
        gt_t = set(ec_at_level(x, lvl) for x in g)
        gl = sorted(gt_t)[0]
        c = preds.get(e, [])
        if not c: continue
        # Take the top-1 (most confident) prediction.
        pl = ec_at_level(c[0][0], lvl)
        yt.append(gl)
        yp.append(pl)
        np_ += 1
        if pl in gt_t:
            nc += 1
    if np_ == 0:
        return {"n_total": nt, "n_predicted": 0, "n_correct": 0,
                "accuracy": 0, "f1_macro": 0, "coverage": 0}
    from sklearn.metrics import f1_score
    f1 = f1_score(yt, yp, labels=sorted(set(yt+yp)),
                  average="macro", zero_division=0)
    return {"n_total": nt, "n_predicted": np_, "n_correct": nc,
            "accuracy": nc/np_, "f1_macro": f1,
            "coverage": np_/nt if nt else 0}
