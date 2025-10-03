# src/data/validate.py
import argparse, sys, re
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

# canonical names we want to end up with
CANON = [
    "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
    "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"
]

# map common variants -> canonical
ALIASES = {
    "fixed_acidity": "fixed acidity",
    "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid",
    "residual_sugar": "residual sugar",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide",
    "sulfur_dioxide_free": "free sulfur dioxide",
    "sulfur_dioxide_total": "total sulfur dioxide",
    "sulphates": "sulphates",
    "sulfates": "sulphates",   # sometimes American spelling
    "ph": "pH",
}

schema = DataFrameSchema(
    {
        "fixed acidity": Column(float, checks=pa.Check.ge(0)),
        "volatile acidity": Column(float, checks=pa.Check.ge(0)),
        "citric acid": Column(float, checks=pa.Check.ge(0)),
        "residual sugar": Column(float, checks=pa.Check.ge(0)),
        "chlorides": Column(float, checks=pa.Check.ge(0)),
        "free sulfur dioxide": Column(float, checks=pa.Check.ge(0)),
        "total sulfur dioxide": Column(float, checks=pa.Check.ge(0)),
        "density": Column(float, checks=pa.Check.ge(0)),
        "pH": Column(float, checks=pa.Check.ge(0)),
        "sulphates": Column(float, checks=pa.Check.ge(0)),
        "alcohol": Column(float, checks=pa.Check.ge(0)),
        "quality": Column(int, checks=pa.Check.between(0, 10)),
    },
    strict=False,
    coerce=True,
)

def normalize(col: str) -> str:
    c = col.strip()
    c = re.sub(r"\s+", " ", c)     # collapse spaces
    # make a key for alias lookup
    key = c.lower().replace(" ", "_").replace("-", "_")
    # special case pH: preserve canonical 'pH' if it's exactly that (case-insensitive)
    if key in ("ph",):
        return "pH"
    # map aliases or return original spacing/case if already canonical
    return ALIASES.get(key, c)

def read_auto(path: str, sep_arg: str | None):
    if sep_arg in (None, "auto"):
        # sniff delimiter
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path, sep=sep_arg)

def main(input_path: str, sep: str | None):
    print(f"[validate] Reading CSV: {input_path} (sep='{sep}')")
    df = read_auto(input_path, sep)

    print("[validate] Original columns:", list(df.columns))

    # Normalize headers
    new_cols = [normalize(c) for c in df.columns]
    df.columns = new_cols
    print("[validate] Normalized columns:", list(df.columns))

    # Check presence of required columns
    missing = [c for c in CANON if c not in df.columns]
    if missing:
        print("\n[validate] ❌ Missing required columns after normalization:")
        for m in missing:
            print("  -", m)
        print("\nTIP: Open the CSV and compare header names. If they differ, we can extend ALIASES.")
        sys.exit(1)

    # Coerce numeric cols manually to be extra safe
    num_cols = [c for c in CANON if c != "quality"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["quality"] = pd.to_numeric(df["quality"], errors="coerce").astype("Int64")

    print("[validate] dtypes after coercion:")
    print(df[CANON].dtypes)

    # Pandera validation (shows full table on error)
    try:
        schema.validate(df[CANON], lazy=True)
        print("[validate] Schema validation passed ✅")
    except pa.errors.SchemaErrors as e:
        print("\n[validate] ❌ Schema validation failed. Details:\n")
        print(e.failure_cases.to_string(index=False))
        sys.exit(1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--sep", type=str, default="auto")  # auto-detect by default
    args = ap.parse_args()
    main(args.input, args.sep)
