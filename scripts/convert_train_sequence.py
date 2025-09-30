#!/usr/bin/env python3
"""
Convert Dataset_and_train_sequence/train_sequence.json into a CSV that
antimicrobial_predictor.py can train on.

Output columns: dna_sequence, protein_sequence, antimicrobial_activity

Usage:
  python scripts/convert_train_sequence.py \
    --in Dataset_and_train_sequence/train_sequence.json \
    --out Dataset_and_train_sequence/antimicrobial_training_data.csv
"""

import argparse
import json
import re
from typing import Any, Dict, Iterable, List

import pandas as pd


def normalize_protein(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    return re.sub(r"[^A-Za-z]", "", seq).upper()


def normalize_dna(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    seq = re.sub(r"[^ATGCatgcNn]", "N", seq)
    return seq.upper()


def to_int_label(value: Any) -> int:
    # Accept 0/1, True/False, 'active'/'inactive', 'yes'/'no'
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        iv = int(value)
        return 1 if iv != 0 else 0
    except Exception:
        s = str(value).strip().lower()
        if s in {"active", "yes", "y", "true", "+", "pos"}:
            return 1
        if s in {"inactive", "no", "n", "false", "-", "neg"}:
            return 0
        return 0


PROTEIN_KEYS = [
    "protein_sequence",
    "peptide_sequence",
    "peptide",
    "sequence",
]

DNA_KEYS = [
    "dna_sequence",
    "microorganism_sequence",
    "microorganism",
    "bacterial_sequence",
    "bacterial_dna",
]

LABEL_KEYS = [
    "antimicrobial_activity",
    "label",
    "active",
    "is_active",
]


def extract_field(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    # case-insensitive fallback
    lower = {k.lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    return None


def convert(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        pep = extract_field(rec, PROTEIN_KEYS)
        dna = extract_field(rec, DNA_KEYS)
        lab = extract_field(rec, LABEL_KEYS)

        pep = normalize_protein(pep)
        dna = normalize_dna(dna)
        lab = to_int_label(lab)

        if not pep or not dna:
            continue

        rows.append({
            "dna_sequence": dna,
            "protein_sequence": pep,
            "antimicrobial_activity": lab,
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", type=str,
                    default="Dataset_and_train_sequence/train_sequence.json")
    ap.add_argument("--out", dest="output_path", type=str,
                    default="Dataset_and_train_sequence/antimicrobial_training_data.csv")
    args = ap.parse_args()

    with open(args.input_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        records = data["data"]
    elif isinstance(data, list):
        records = data
    else:
        raise SystemExit("Unsupported JSON structure: expected list or {data: [...]}.")

    df = convert(records)
    if df.empty:
        raise SystemExit("No valid records extracted (missing sequences?).")

    df.to_csv(args.output_path, index=False)
    print(f"Wrote {len(df)} rows to {args.output_path}")


if __name__ == "__main__":
    main()


