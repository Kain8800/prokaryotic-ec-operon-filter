*English · [Русский](SETUP_ENV.ru.md)*

# Setting up the `clean_env` environment

This procedure reproduces the EC-number prediction pipeline environment on a fresh
machine (laptop or cluster). `venv` + `pip` only, no conda/poetry/docker.

## Interpreter requirements

Requires **Python 3.10** (3.9 is acceptable). This is the ceiling: `numpy 1.22.3`
and `pandas 1.4.2` have no wheels for Python 3.11 and newer and do not build from
source on newer interpreters. Check that the right Python is available:

```bash
python3.10 --version    # expected: Python 3.10.x
```

If the cluster uses a module system, load the right version first, e.g.
`module load python/3.10` (get the exact module name via `module avail python`).

---

## Installation

```bash
python3.10 -m venv clean_env
source clean_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## External dependencies outside pip

Installed separately; they are not in `requirements.txt`:

- **MMseqs2** — an external CLI for homology stratification. You need a build for
  the compute node's instruction set: on nodes without AVX2 the AVX2 binary
  crashes with `rc=-4` (SIGILL). Use a **static SSE4.1 build**; the path to the
  binary is set via the `MMSEQS_BIN` environment variable
  (`export MMSEQS_BIN=/path/to/mmseqs`, see README).
- **TensorFlow** — *optional*. Needed only for fast reading of the ProteInfer
  TFRecord. Without it, `7_EVALUATION.py` falls back to a built-in manual protobuf
  parse, so TF is not added to the base environment (it is heavy). It can be
  installed separately with a Python 3.10-compatible version if desired.

---

## Environment check

After installation:

```bash
python - <<'PY'
import numpy, pandas, pyarrow, matplotlib, sklearn, scipy
for m in (numpy, pandas, pyarrow, matplotlib, sklearn, scipy):
    print(f"{m.__name__:12s} {m.__version__}")
PY
```

Expected output (transitive versions may differ slightly; the key ones are pinned):

```
numpy        1.22.3
pandas       1.4.2
pyarrow      20.0.0
matplotlib   3.7.0
sklearn      1.2.0
scipy        1.11.4
```

---

## A note on versions

The thesis (section 2.1, the ODB stage) lists `pandas 1.3.5` and `numpy 1.21.5`
for preprocessing — the versions from an early stage of the work. The final
consolidated `clean_env` environment uses `pandas 1.4.2` / `numpy 1.22.3`; on the
ODB-stage results (`explode`, `dropna`, `drop_duplicates`) the difference between
these minor versions has no effect. This procedure is built for `clean_env`.
