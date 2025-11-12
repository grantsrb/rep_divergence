"""
Find proportional duplicate non-zero rows (and optionally columns) in 2-D weight tensors of a Hugging Face model.

Proportional duplicates: rows r_i, r_j such that r_i ≈ c * r_j for some scalar c
(negative c allowed by default). We normalize each row by its largest-magnitude
entry ("anchor" normalization) to canonicalize directions, then group by quantized
canonical directions to detect (near-)proportionality.

Usage:
  python find_proportional_dupes.py --model gpt2
  python find_proportional_dupes.py --model meta-llama/Llama-3-8b --tol 1e-6
  python find_proportional_dupes.py --model gpt2 --check_columns --positive-only
  python find_proportional_dupes.py --model /path/to/local --json-out report.json
"""

import argparse
from collections import defaultdict
import json
import math
import torch

# ------------------------------ HF loader ------------------------------

def load_hf_model(model_id, trust_remote_code=False, revision=None, dtype=torch.float32):
    from transformers import AutoModel, AutoModelForCausalLM
    kwargs = dict(
        revision=revision,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    try:
        model = AutoModel.from_pretrained(model_id, **kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model

# ------------------------- Proportional grouping ------------------------

def _anchor_normalize_rows(rows: torch.Tensor, positive_only: bool) -> torch.Tensor:
    """
    Canonicalize each non-zero row by dividing by its anchor element: the entry
    with maximum absolute value. This maps any proportional vectors (including
    negative multiples if positive_only=False) to (approximately) the same direction.

    If positive_only=True: flip sign so the anchor becomes +|anchor|.
    If positive_only=False: keep the original sign of the anchor so -r is distinct
    from r unless tol groups them later.
    """
    # rows: (k, n), none are all-zero
    abs_rows = rows.abs()
    anchor_pos = abs_rows.argmax(dim=1)                   # (k,)
    anchors = rows[torch.arange(rows.size(0)), anchor_pos]  # (k,)

    if positive_only:
        # Force anchor to be positive by flipping sign if needed
        signs = torch.sign(anchors)
        signs[signs == 0] = 1.0
        rows = rows * signs.unsqueeze(1)
        anchors = anchors.abs()

    # Safe divide (anchors are max-abs entries, so nonzero if row nonzero)
    canon = rows / anchors.unsqueeze(1)
    return canon


def _quantize(mat: torch.Tensor, tol: float) -> torch.Tensor:
    """
    Quantize a matrix for approximate grouping. tol=0 -> exact matching.
    Otherwise round(mat / tol) to integers.
    """
    if tol <= 0:
        return mat  # exact mode (torch.unique on float rows is okay if exactly equal)
    scale = 1.0 / tol
    return torch.round(mat * scale).to(torch.int64)


def find_proportional_duplicate_rows(W: torch.Tensor,
                                     tol: float = 0.0,
                                     positive_only: bool = False):
    """
    Return a list of groups of row indices that are proportional duplicates
    among non-zero rows. Each group has length >= 2.

    - tol: numeric tolerance for near-proportional (0 => exact).
    - positive_only: if True, treats r and -r as the same direction only if
      their anchors are flipped to positive (i.e., collapse sign).
    """
    assert W.ndim == 2
    with torch.no_grad():
        R = W.detach().cpu()
        # filter non-zero rows
        nz_mask = (R.abs().sum(dim=1) > 0)
        idx_all = torch.nonzero(nz_mask, as_tuple=False).flatten()
        if idx_all.numel() == 0:
            return []

        Rn = R[idx_all]
        canon = _anchor_normalize_rows(Rn.clone(), positive_only=positive_only)
        key = _quantize(canon, tol=tol)

        # Group by canonical (quantized) direction
        uniq, inv, counts = torch.unique(key, dim=0, return_inverse=True, return_counts=True)

        groups = []
        for u_idx, c in enumerate(counts.tolist()):
            if c > 1:
                members = idx_all[(inv == u_idx)].tolist()
                groups.append(members)

        return groups


def find_proportional_duplicate_cols(W: torch.Tensor,
                                     tol: float = 0.0,
                                     positive_only: bool = False):
    """Same as rows but across non-zero columns."""
    return find_proportional_duplicate_rows(W.t(), tol=tol, positive_only=positive_only)

# ----------------------------- Reporting utils -----------------------------

def scalar_multipliers_against_ref(R: torch.Tensor, group_idxs):
    """
    For a group of row indices (in the *original* tensor R), compute least-squares
    scalars c_i such that R[i] ≈ c_i * R[ref] for a chosen reference row (first).
    Returns dict: {"ref": ref_idx, "members": [(idx, c_i, rel_err), ...]}
    rel_err = ||R[i] - c_i R[ref]|| / (||R[i]|| + 1e-12)
    """
    with torch.no_grad():
        ref = group_idxs[0]
        r_ref = R[ref]
        denom = (r_ref @ r_ref).item()
        out = {"ref": ref, "members": []}
        for idx in group_idxs:
            r = R[idx]
            if denom > 0:
                c = (r @ r_ref).item() / denom
            else:
                c = float("nan")
            rel_err = (r - c * r_ref).norm().item() / (r.norm().item() + 1e-12)
            out["members"].append((idx, c, rel_err))
        return out

# ---------------------------------- Main -----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path (e.g., gpt2)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tol", type=float, default=0.0, help="tolerance for near-proportional grouping (0=exact)")
    ap.add_argument("--positive-only", action="store_true",
                    help="treat negative multiples as the same direction by flipping anchor sign to positive")
    ap.add_argument("--check_columns", action="store_true", help="also check proportional duplicate columns")
    ap.add_argument("--json-out", default=None, help="optional path to save JSON report")
    ap.add_argument("--dtype", default="float32", choices=["float16", "float32", "bfloat16"])
    args = ap.parse_args()

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    model = load_hf_model(args.model, trust_remote_code=args.trust_remote_code,
                          revision=args.revision, dtype=dtype_map[args.dtype])

    report = {
        "model": args.model,
        "tol": args.tol,
        "positive_only": args.positive_only,
        "rows": {},
        "cols": {} if args.check_columns else None,
        "details": {}
    }

    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        try:
            row_groups = find_proportional_duplicate_rows(
                p, tol=args.tol, positive_only=args.positive_only
            )
        except RuntimeError as e:
            print(f"[WARN] Skipping {name} (rows): {e}")
            row_groups = []

        if row_groups:
            report["rows"][name] = row_groups

            # Add per-group scalar estimates and errors for inspection
            R = p.detach().cpu()
            detail_groups = []
            for g in row_groups:
                detail_groups.append(scalar_multipliers_against_ref(R, g))
            report["details"][f"{name}::rows"] = detail_groups

        if args.check_columns:
            try:
                col_groups = find_proportional_duplicate_cols(
                    p, tol=args.tol, positive_only=args.positive_only
                )
            except RuntimeError as e:
                print(f"[WARN] Skipping {name} (cols): {e}")
                col_groups = []

            if col_groups:
                report["cols"][name] = col_groups
                C = p.detach().cpu().t()
                detail_groups = []
                for g in col_groups:
                    detail_groups.append(scalar_multipliers_against_ref(C, g))
                report["details"][f"{name}::cols"] = detail_groups

    # Pretty print
    print("\n=== Proportional Duplicate Non-Zero ROWS ===")
    if not report["rows"]:
        print("No proportional duplicate non-zero rows found.")
    else:
        for pname, groups in report["rows"].items():
            print(f"\n[pname: {pname}] proportional row groups (indices):")
            for g in groups:
                print("  -", g)

    if args.check_columns:
        print("\n=== Proportional Duplicate Non-Zero COLUMNS ===")
        if not report["cols"]:
            print("No proportional duplicate non-zero columns found.")
        else:
            for pname, groups in report["cols"].items():
                print(f"\n[pname: {pname}] proportional column groups (indices):")
                for g in groups:
                    print("  -", g)

    if args.json_out:
        # Convert tuples to lists for JSON
        serializable = json.loads(json.dumps(report, default=lambda o: o))
        with open(args.json_out, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nSaved JSON report to: {args.json_out}")


if __name__ == "__main__":
    main()
