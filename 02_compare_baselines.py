#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_compare_baselines.py
=======================
Side-by-side comparison of the zero-baseline and mean-baseline runs.

The comparison IS the result. The zero baseline is not simply worse -- showing
how and why it fails is a reportable methodological finding, and one that
applies to any IG analysis of single-cell expression data.

Four questions:
  1. Baseline health -- what fraction of cells have delta < 0? A baseline that
     projects higher on a class axis than real cells of that class is
     off-manifold and the attributions inherit that.
  2. Housekeeping dominance -- how much attribution mass sits on mitochondrial
     and ribosomal genes, against their share of the panel?
  3. Reproducibility -- cross-split agreement, measured as enrichment over
     chance rather than as a raw 5/5 count.
  4. Content -- do the reproducible genes change from housekeeping to immune?

CPU only. Run after both baselines have finished.

    python 02_compare_baselines.py
"""

from __future__ import print_function

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RES = "../results/interpretability"
FIG = "../figures/interpretability"
HOUSEKEEPING = re.compile(r"^(MT-|MTRNR|RPL|RPS|RPLP)")


def log(m=""):
    print(m, flush=True)


def banner(m):
    log("\n" + "=" * 78)
    log(m)
    log("=" * 78)


def read_csv(base, name):
    p = os.path.join(RES, base, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=["zero", "mean"])
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(os.path.join(FIG, "comparison"), exist_ok=True)
    available = [b for b in args.baselines
                 if os.path.isdir(os.path.join(RES, b))]
    missing = [b for b in args.baselines if b not in available]
    if missing:
        log("missing runs: {0}".format(missing))
    if not available:
        log("nothing to compare")
        return 1

    # ---- 1. baseline health ------------------------------------------------
    banner("1. BASELINE HEALTH  --  fraction of cells with delta < 0")
    log("delta < 0 means the baseline projects HIGHER on a class axis than the")
    log("real cell does. Those cells are being explained relative to a point")
    log("that is more 'characteristic' of their class than they are.\n")
    health = {}
    for b in available:
        dd = read_csv(b, "delta_diagnostics.csv")
        if dd is None:
            continue
        health[b] = dd.groupby("antigen")["frac_delta_negative"].mean()
    if health:
        hf = pd.DataFrame(health)
        log(hf.round(4).to_string())
        log("\noverall:")
        for b in hf.columns:
            log("  {0:>5}: {1:.1%}".format(b, hf[b].mean()))
        hf.to_csv(os.path.join(RES, "comparison_delta_health.csv"))

    # ---- 2. completeness ---------------------------------------------------
    banner("2. COMPLETENESS  --  is the integration still accurate?")
    for b in available:
        c = read_csv(b, "completeness_per_split.csv")
        if c is None:
            continue
        log("{0:>5}: median {1:.5f}   p90 {2:.5f}   mean {3:.5f}".format(
            b, c["median_rel_err"].median(), c["p90_rel_err"].median(),
            c["mean_rel_err"].mean()))
    log("\nReport the MEDIAN. Relative error is undefined for cells whose delta")
    log("is ~0, and a handful of those inflate the mean by an order of magnitude.")

    # ---- 3. housekeeping dominance ----------------------------------------
    banner("3. HOUSEKEEPING DOMINANCE")
    hk_rows = []
    for b in available:
        d = os.path.join(RES, b)
        for f in sorted(os.listdir(d)):
            if not f.startswith("attr_split") or not f.endswith(".npz"):
                continue
            z = np.load(os.path.join(d, f), allow_pickle=True)
            genes = z["genes"].astype(str)
            cls = z["classes"].astype(str)
            hk = np.array([bool(HOUSEKEEPING.match(g)) for g in genes])
            rna = z["rna"]
            # Accumulate in row chunks. The real matrix is 61,227 x 5,000, which
            # is 2.45 GB as float64 -- materialising it would risk an OOM kill on
            # the login node, and this script is meant to run there.
            tot = {c: 0.0 for c in np.unique(cls)}
            hks = {c: 0.0 for c in np.unique(cls)}
            for i in range(0, rna.shape[0], 4096):
                chunk = np.abs(rna[i:i + 4096].astype(np.float64))
                sub = cls[i:i + 4096]
                for c in np.unique(sub):
                    m = sub == c
                    tot[c] += float(chunk[m].sum())
                    hks[c] += float(chunk[m][:, hk].sum())
                del chunk
            for c in tot:
                hk_rows.append({
                    "baseline": b, "split": f,
                    "antigen": str(c).replace("_binder", ""),
                    "hk_fraction": (hks[c] / tot[c]) if tot[c] else np.nan,
                    "hk_share_of_panel": float(hk.mean())})
            del rna, z
    if hk_rows:
        hk = pd.DataFrame(hk_rows)
        hk.to_csv(os.path.join(RES, "comparison_housekeeping.csv"), index=False)
        piv = hk.pivot_table(index="antigen", columns="baseline",
                             values="hk_fraction")
        share = hk["hk_share_of_panel"].iloc[0]
        log("Share of |RNA attribution| on mitochondrial + ribosomal genes")
        log("(these are {0:.1%} of the gene panel):\n".format(share))
        log(piv.round(4).to_string())
        log("\nenrichment over their share of the panel:")
        for b in piv.columns:
            log("  {0:>5}: {1:.1f}x".format(b, piv[b].mean() / share))

        fig, ax = plt.subplots(figsize=(7.5, 4))
        piv.plot.barh(ax=ax)
        ax.axvline(share, color="k", ls="--", lw=1,
                   label="share of panel ({0:.1%})".format(share))
        ax.set_xlabel("fraction of |RNA attribution| on housekeeping genes")
        ax.set_title("Housekeeping dominance by baseline", fontsize=10)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "comparison", "housekeeping_by_baseline.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ---- 4. reproducibility ------------------------------------------------
    banner("4. REPRODUCIBILITY  --  cross-split agreement vs chance")
    for b in available:
        a = read_csv(b, "cross_split_agreement.csv")
        if a is None:
            continue
        log("\n{0} baseline:".format(b))
        log(a.round(4).to_string(index=False))

    # ---- 5. content --------------------------------------------------------
    banner("5. CONTENT  --  what are the reproducible genes?")
    for b in available:
        s = read_csv(b, "cross_split_summary.csv")
        if s is None or not len(s):
            log("\n{0}: no reproducible genes".format(b))
            continue
        s["housekeeping"] = s["gene"].astype(str).str.match(HOUSEKEEPING)
        n_hk = int(s["housekeeping"].sum())
        n_tcr = int(s["tcr_locus"].sum()) if "tcr_locus" in s else 0
        log("\n{0} baseline: {1} reproducible gene-class pairs".format(b, len(s)))
        log("  housekeeping : {0} ({1:.0%})".format(n_hk, n_hk / max(len(s), 1)))
        log("  TCR locus    : {0}".format(n_tcr))
        log("  other        : {0}".format(len(s) - n_hk - n_tcr))
        other = s[~s["housekeeping"]]
        if len(other):
            log("\n  non-housekeeping genes:")
            log(other.head(25).to_string(index=False))

    # ---- verdict -----------------------------------------------------------
    banner("VERDICT")
    if len(available) < 2:
        log("Only one baseline available -- rerun the other to compare.")
        return 0
    log("Judge the mean baseline a success if, relative to zero:")
    log("  * delta < 0 collapses toward ~0%  (baseline now on-manifold)")
    log("  * housekeeping enrichment drops sharply")
    log("  * the reproducible gene list shifts from MT-/RPS-/RPL- to immune genes")
    log("  * completeness medians stay comparable (integration still accurate)")
    log("")
    log("If housekeeping dominance persists under the mean baseline, the")
    log("conclusion is stronger and different: the model's latent placement")
    log("genuinely depends on library composition, not on antigen-specific")
    log("expression. That is a finding about mvTCR, not about the method.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
