#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_run_analysis.py
==================
Full IG interpretability analysis for ONE choice of RNA baseline.

Replaces 02_ig_analysis.ipynb. A plain script rather than a notebook because
nbconvert buffers all output into the .ipynb and writes it only at the end, so
a crash in the last cell loses every printed number from a 3.5-hour run. Here
every table and figure is written to disk as it is produced.

    --baseline zero    x' = 0. Attribution is x * grad, so the highest-expressed
                       genes (mitochondrial, ribosomal) dominate mechanically.
                       Also lands off-manifold: in the first run 30.4% of cells
                       had F(x) < F(x'), i.e. an empty cell looked MORE
                       characteristic of the class than a real one.
    --baseline mean    x' = mean log-normalised expression over training cells.
                       Asks "what makes this cell unusual" instead of "what
                       makes this cell non-empty".

Run both and compare with 02_compare_baselines.py.

Fixes carried over from the first run:
  * normalisation divides each class by its own median |delta| (one positive
    constant per class per split). The first run divided each cell by its own
    SIGNED delta, which inverted every attribution for the 30% of cells with
    delta < 0 and let one boundary cell dominate a 26,322-cell average
  * cross-split reporting shows the full agreement distribution with chance
    expectation, never just the 5/5 count, and guards every empty DataFrame
  * save_split_attributions persists everything, including the cell-weighted
    arrays that were lost last time
  * per-split results are written immediately, so a late failure costs nothing

Run:
    python 01_run_analysis.py --baseline zero
    python 01_run_analysis.py --baseline mean
"""

from __future__ import print_function

import argparse
import gc
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.append("../mvTCR/")
import attribution_core as ac


# ==============================================================================
def log(msg=""):
    print(msg, flush=True)


def banner(msg):
    log("\n" + "=" * 78)
    log(msg)
    log("=" * 78)


def row_labels(df, counts):
    return ["{0}  (n={1})".format(str(i).replace("_binder", ""),
                                  int(counts.get(i, 0))) for i in df.index]


# ==============================================================================
def run_split(split, cfg):
    """Attribute every cell of one split. Returns arrays + metadata."""
    ad = ac.load_split(split, donor=cfg["donor"])
    mdl = ac.load_model_for_split(ad, split, donor=cfg["donor"],
                                  model_name=cfg["model"])
    wr = ac.JointLatentWrapper(mdl)

    lat = ac.compute_latents(wr, ad, batch_size=512)
    cls = ad.obs["binding_name"].astype(str).values
    clo = ad.obs["clonotype"].astype(str).values
    st = ad.obs["set"].astype(str).values
    pmask = st == cfg["proto_set"]

    # The mean baseline is computed from TRAINING cells only, so held-out data
    # never influences the reference point.
    base_vec = None
    if cfg["baseline"] == "mean":
        base_vec = ac.mean_expression_baseline(ad, mask=pmask)
        log("  mean baseline: {0} genes, mean {1:.4f}, max {2:.4f}".format(
            base_vec.shape[0], float(base_vec.mean()), float(base_vec.max())))

    pb = ac.compute_prototypes(lat[pmask], cls[pmask], clo[pmask], balanced=True)
    axes_bal = {c: ac.contrastive_axis(pb, c) for c in pb}
    axes_cw = None
    if cfg["with_cellweighted"]:
        pc = ac.compute_prototypes(lat[pmask], cls[pmask], clo[pmask],
                                   balanced=False)
        axes_cw = {c: ac.contrastive_axis(pc, c) for c in pc}

    n, G = ad.shape
    L = mdl.params_tcr["max_tcr_length"]
    A = {"rna": np.zeros((n, G), np.float32),
         "alpha": np.zeros((n, L), np.float32),
         "beta": np.zeros((n, L), np.float32),
         "delta": np.zeros(n, np.float32),
         "total": np.zeros(n, np.float32)}
    cw_rna = np.zeros((n, G), np.float32) if cfg["with_cellweighted"] else None
    cw_delta = np.zeros(n, np.float32) if cfg["with_cellweighted"] else None

    t0, done = time.time(), 0
    for rna, tcr, tlen, cond, idx in ac.iter_batches(ad, cfg["cell_batch"]):
        for c in np.unique(cls[idx]):
            sel = np.where(cls[idx] == c)[0]
            if not len(sel):
                continue
            sl = torch.as_tensor(sel)
            args = (rna[sl], tcr[sl], tlen[sl],
                    None if cond is None else cond[sl])
            gi = idx[sel]

            r = ac.integrated_gradients(
                wr, *args, u=axes_bal[c], n_steps=cfg["n_steps"],
                step_chunk=cfg["step_chunk"], rna_baseline=base_vec)
            A["rna"][gi] = r["attr_rna"]
            A["alpha"][gi] = r["attr_alpha"]
            A["beta"][gi] = r["attr_beta"]
            A["delta"][gi] = r["delta"]
            A["total"][gi] = r["attr_total"]

            if cfg["with_cellweighted"]:
                r2 = ac.integrated_gradients(
                    wr, *args, u=axes_cw[c], n_steps=cfg["n_steps"],
                    step_chunk=cfg["step_chunk"], rna_baseline=base_vec)
                cw_rna[gi] = r2["attr_rna"]
                cw_delta[gi] = r2["delta"]

        done += len(idx)
        if done % (cfg["cell_batch"] * 60) == 0:
            el = time.time() - t0
            log("    {0}/{1}  {2:.1f} min elapsed, ~{3:.1f} min left".format(
                done, n, el / 60, el / done * (n - done) / 60))

    err = np.abs(A["total"] - A["delta"]) / np.maximum(np.abs(A["delta"]), 1e-8)
    log("  split {0}: {1} cells in {2:.1f} min".format(
        split, n, (time.time() - t0) / 60))
    log("    completeness  mean {0:.4f}  median {1:.5f}  p90 {2:.4f}".format(
        err.mean(), np.median(err), np.percentile(err, 90)))
    log("    delta < 0 in {0:.1%} of cells".format((A["delta"] < 0).mean()))
    # The normalisation divisor: one median |delta| per class. Printed because
    # it is the only quantity standing between raw attributions and the
    # cross-split comparison, and because a class whose divisor is orders of
    # magnitude off the others is worth noticing before the figures are drawn.
    sc = ac.class_median_scale(A["delta"], cls)
    divisors = {str(c).replace("_binder", ""): round(float(sc[cls == c][0]), 4)
                for c in np.unique(cls)}
    log("    normalisation divisors (median |delta| per class):")
    for k, v in sorted(divisors.items(), key=lambda kv: -kv[1]):
        log("      {0:<34} {1}".format(k, v))

    return {"split": split, "attr": A, "classes": cls, "sets": st,
            "clonotypes": clo, "genes": list(ad.var_names),
            "tokens_alpha": np.asarray(ad.obsm["alpha_seq"]),
            "tokens_beta": np.asarray(ad.obsm["beta_seq"]),
            "attr_cellweighted_rna": cw_rna, "delta_cellweighted": cw_delta,
            "completeness": err, "n_cells": n, "baseline": cfg["baseline"],
            "mean_baseline_vec": base_vec}


# ==============================================================================
def aggregate(res, restrict_set=None):
    """Class-level aggregates.

    Normalisation happens HERE -- after IG, before any cross-split comparison.
    Each class is divided by its own median |delta|, so attributions become
    dimensionless and therefore comparable between separately trained splits.
    The saved .npz files stay raw, so alternative schemes can be tested later
    without touching the GPU.
    """
    m = (np.ones(res["n_cells"], bool) if restrict_set is None
         else (res["sets"] == restrict_set))
    A, cls = res["attr"], res["classes"][m]
    d = A["delta"][m]

    g_signed, g_abs = ac.aggregate_rna(A["rna"][m], cls, res["genes"],
                                       normalise_by=d)
    out = {"gene_signed": g_signed, "gene_abs": g_abs,
           "tcr_mass": ac.tcr_gene_mass(A["rna"][m], cls, res["genes"]),
           "delta_diag": ac.delta_diagnostics(d, cls),
           "n_cells": int(m.sum()),
           "class_counts": pd.Series(cls).value_counts()}

    L = A["alpha"].shape[1]
    cterm = ["-{0}".format(i) for i in range(L, 0, -1)]
    for chain, key in [("alpha", "tokens_alpha"), ("beta", "tokens_beta")]:
        pos, tok = A[chain][m], res[key][m]
        sN, aN = ac.aggregate_positions(pos, tok, cls, normalise_by=d)
        anc, anc_tok = ac.anchor_to_cterm(pos, tok)
        sC, aC = ac.aggregate_positions(anc, anc_tok, cls, normalise_by=d,
                                        column_labels=cterm)
        out[chain + "_signed"], out[chain + "_abs"] = sN, aN
        out[chain + "_signed_cterm"], out[chain + "_abs_cterm"] = sC, aC

    if res.get("attr_cellweighted_rna") is not None:
        cs, ca = ac.aggregate_rna(res["attr_cellweighted_rna"][m], cls,
                                  res["genes"],                                   normalise_by=res["delta_cellweighted"][m])
        out["cw_gene_signed"], out["cw_gene_abs"] = cs, ca
    return out


# ==============================================================================
def fig_genes(agg, fig_dir, split, top_n, tag=""):
    df_abs, df_signed = agg["gene_abs"], agg["gene_signed"]
    keep = []
    for c in df_abs.index:
        keep += list(df_abs.loc[c].sort_values(ascending=False).head(top_n).index)
    keep = list(dict.fromkeys(keep))
    M = df_signed[keep]
    is_tcr = ac.identify_tcr_genes(keep)
    v = np.nanmax(np.abs(M.values))
    if not np.isfinite(v) or v == 0:
        v = 1.0
    fig, ax = plt.subplots(figsize=(max(10, len(keep) * 0.22),
                                    0.45 * len(M) + 2.4))
    im = ax.imshow(M.values, aspect="auto", cmap="RdBu_r",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v))
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels(["{0} *".format(g) if t else g
                        for g, t in zip(keep, is_tcr)], rotation=90, fontsize=7)
    for tick, t in zip(ax.get_xticklabels(), is_tcr):
        if t:
            tick.set_color("crimson")
            tick.set_fontweight("bold")
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels(row_labels(M, agg["class_counts"]), fontsize=8)
    ax.set_title("Top {0} genes per antigen, split {1}{2}\n"
                 "(red * = TCR locus V/D/J/C)".format(top_n, split, tag),
                 fontsize=10)
    plt.colorbar(im, ax=ax, label="mean signed attribution (class-median normalised)",
                 fraction=0.02, pad=0.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_genes_split{0}.png".format(split)),
                dpi=150, bbox_inches="tight")
    plt.close()
    return keep


def fig_positions(agg, fig_dir, split, chain):
    dN, dC = agg[chain + "_signed"], agg[chain + "_signed_cterm"]
    v = np.nanmax([np.nanmax(np.abs(dN.values)), np.nanmax(np.abs(dC.values))])
    if not np.isfinite(v) or v == 0:
        v = 1.0
    norm = TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v)
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.45 * len(dN) + 2.4))
    for ax, M, ttl, xlab, lab in [
            (axes[0], dN, "anchored at the N-terminal C",
             "position from the start of CDR3", True),
            (axes[1], dC, "anchored at the C-terminal F/W",
             "position from the end of CDR3", False)]:
        im = ax.imshow(M.values, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels(list(M.columns), fontsize=6, rotation=90)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_yticks(range(len(M)))
        ax.set_yticklabels(row_labels(M, agg["class_counts"]) if lab else [],
                           fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.set_facecolor("0.9")
    fig.suptitle("CDR3{0} positional attribution, split {1}  "
                 "(grey = no cell in that class reaches this position)".format(
                     chain[0], split), fontsize=11, y=1.02)
    fig.colorbar(im, ax=axes, label="mean signed attribution",
                 fraction=0.015, pad=0.01)
    n = {"alpha": "fig2_cdr3a", "beta": "fig3_cdr3b"}[chain]
    plt.savefig(os.path.join(fig_dir, "{0}_split{1}.png".format(n, split)),
                dpi=150, bbox_inches="tight")
    plt.close()


def fig_tcr_mass(agg, fig_dir, n_genes):
    tm = agg["tcr_mass"].sort_values("tcr_locus_fraction", ascending=False)
    n_tcr = agg["tcr_mass"].attrs.get("n_tcr_genes", 0)
    expected = float(n_tcr) / float(n_genes)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.barh(range(len(tm)), tm["tcr_locus_fraction"].values, color="indianred")
    ax.axvline(expected, color="k", ls="--", lw=1,
               label="proportional share ({0:.3f})".format(expected))
    ax.set_yticks(range(len(tm)))
    ax.set_yticklabels([str(i).replace("_binder", "") for i in tm.index],
                       fontsize=8)
    ax.set_xlabel("fraction of |RNA attribution| on TCR locus genes")
    ax.set_title("TCR-locus share ({0} of {1} genes)".format(n_tcr, n_genes))
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "tcr_locus_fraction.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


def fig_agreement(dist, fig_dir, n_splits):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(dist.index, dist["observed"], color="steelblue", label="observed")
    ax.plot(dist.index, np.maximum(dist["expected_by_chance"], 1e-3), "o--",
            color="crimson", label="expected by chance")
    ax.set_yscale("log")
    ax.set_xticks(list(dist.index))
    ax.set_xlabel("splits (of {0}) placing the gene in the class top-N".format(
        n_splits))
    ax.set_ylabel("gene-class pairs (log)")
    ax.set_title("Cross-split agreement vs chance", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig6_agreement_vs_chance.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", choices=["zero", "mean"], required=True)
    ap.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--step-chunk", type=int, default=8)
    ap.add_argument("--cell-batch", type=int, default=64)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--donor", default="None")
    ap.add_argument("--model", default="moe")
    ap.add_argument("--proto-set", default="train")
    ap.add_argument("--restrict-set", default=None)
    ap.add_argument("--no-cellweighted", action="store_true")
    ap.add_argument("--disable-tf32", action="store_true", default=True)
    args = ap.parse_args()

    cfg = {"baseline": args.baseline, "donor": args.donor, "model": args.model,
           "n_steps": args.n_steps, "step_chunk": args.step_chunk,
           "cell_batch": args.cell_batch, "proto_set": args.proto_set,
           "with_cellweighted": not args.no_cellweighted}

    OUT = "../results/interpretability/{0}".format(args.baseline)
    FIG = "../figures/interpretability/{0}".format(args.baseline)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    if torch.cuda.is_available() and args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    banner("mvTCR IG analysis  |  baseline = {0}".format(args.baseline.upper()))
    log("device      : {0}".format(
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    log("splits      : {0}".format(args.splits))
    log("steps       : {0}   cell batch {1}   step chunk {2}".format(
        args.n_steps, args.cell_batch, args.step_chunk))
    log("cell-weighted second pass: {0}".format(cfg["with_cellweighted"]))
    log("output      : {0}".format(OUT))
    if args.baseline == "zero":
        log("\nNOTE: the zero baseline is retained for comparison only. It makes")
        log("attribution proportional to raw expression, and in the first run it")
        log("put F(baseline) above F(x) for 30.4% of cells.")

    all_agg, comp_rows, delta_rows = {}, [], []
    t_start = time.time()

    for s in args.splits:
        banner("SPLIT {0}".format(s))
        try:
            res = run_split(s, cfg)
        except Exception:
            log("SPLIT {0} FAILED:".format(s))
            log(traceback.format_exc())
            continue

        # Save immediately. A late crash must never cost completed compute.
        ac.save_split_attributions(
            os.path.join(OUT, "attr_split{0}.npz".format(s)), res)

        agg = aggregate(res, restrict_set=args.restrict_set)
        agg["completeness_mean"] = float(res["completeness"].mean())
        agg["completeness_median"] = float(np.median(res["completeness"]))
        all_agg[s] = agg

        err = res["completeness"]
        comp_rows.append({"split": s, "n_cells": res["n_cells"],
                          "mean_rel_err": float(err.mean()),
                          "median_rel_err": float(np.median(err)),
                          "p90_rel_err": float(np.percentile(err, 90)),
                          "max_rel_err": float(err.max()),
                          "frac_delta_negative": float((res["attr"]["delta"] < 0).mean())})
        dd = agg["delta_diag"].reset_index()
        dd["split"] = s
        delta_rows.append(dd)

        if s == args.splits[0]:
            fig_genes(agg, FIG, s, args.top_n)
            fig_positions(agg, FIG, s, "alpha")
            fig_positions(agg, FIG, s, "beta")
            fig_tcr_mass(agg, FIG, len(res["genes"]))
            agg["gene_abs"].to_csv(os.path.join(OUT, "gene_abs_split{0}.csv".format(s)))
            agg["gene_signed"].to_csv(os.path.join(OUT, "gene_signed_split{0}.csv".format(s)))

        del res
        gc.collect()

    if not all_agg:
        log("\nNo split completed. Nothing to report.")
        return 1

    # ---- completeness ------------------------------------------------------
    banner("COMPLETENESS")
    comp = pd.DataFrame(comp_rows).set_index("split")
    comp.to_csv(os.path.join(OUT, "completeness_per_split.csv"))
    log(comp.round(5).to_string())
    log("\nMEDIAN is the statistic to report: relative error is meaningless for")
    log("cells whose delta is ~0, and those few cells dominate the mean.")
    log("  median of per-split medians : {0:.5f}  ({1:.3f} %)".format(
        comp["median_rel_err"].median(), 100 * comp["median_rel_err"].median()))
    log("  mean of per-split means     : {0:.5f}  ({1:.3f} %)".format(
        comp["mean_rel_err"].mean(), 100 * comp["mean_rel_err"].mean()))

    banner("BASELINE HEALTH  (fraction of cells with delta < 0)")
    dd_all = pd.concat(delta_rows)
    dd_all.to_csv(os.path.join(OUT, "delta_diagnostics.csv"), index=False)
    piv = dd_all.pivot_table(index="antigen", columns="split",
                             values="frac_delta_negative")
    log(piv.round(4).to_string())
    overall_neg = float(dd_all["frac_delta_negative"].mean())
    log("\noverall delta < 0: {0:.1%}".format(overall_neg))
    log("A healthy baseline gives a small number here. delta < 0 means the")
    log("baseline projects HIGHER on the class axis than the real cell does.")

    # ---- cross-split -------------------------------------------------------
    banner("CROSS-SPLIT AGREEMENT")
    splits_done = sorted(all_agg)
    stab = ac.cross_split_stability([all_agg[s]["gene_abs"] for s in splits_done],
                                    top_n=args.top_n)
    stab.to_csv(os.path.join(OUT, "cross_split_stability.csv"), index=False)

    n_genes = all_agg[splits_done[0]]["gene_abs"].shape[1]
    n_cls = all_agg[splits_done[0]]["gene_abs"].shape[0]
    dist = ac.agreement_distribution(stab, len(splits_done), n_genes=n_genes,
                                     top_n=args.top_n, n_classes=n_cls)
    dist.to_csv(os.path.join(OUT, "cross_split_agreement.csv"))
    log(dist.round(4).to_string())
    log("\nEnrichment, not the raw 5/5 count, is the meaningful number: under")
    log("random ranking the expected count at 5/5 is ~1e-8, so observing zero")
    log("there says nothing either way.")
    fig_agreement(dist, FIG, len(splits_done))

    best = int(stab["n_splits_in_top"].max()) if len(stab) else 0
    log("\nhighest agreement reached: {0}/{1}".format(best, len(splits_done)))

    # ---- per-gene summary, with every empty case guarded -------------------
    cols = ["antigen", "gene", "n_splits_in_top", "mean_abs", "sd_abs", "cv",
            "mean_signed", "tcr_locus"]
    rows = []
    if best >= 1:
        for _, r in stab[stab["n_splits_in_top"] >= max(best, 2)].iterrows():
            vals = [all_agg[s]["gene_abs"].loc[r["antigen"], r["gene"]]
                    for s in splits_done
                    if r["gene"] in all_agg[s]["gene_abs"].columns
                    and r["antigen"] in all_agg[s]["gene_abs"].index]
            sg = [all_agg[s]["gene_signed"].loc[r["antigen"], r["gene"]]
                  for s in splits_done
                  if r["gene"] in all_agg[s]["gene_signed"].columns
                  and r["antigen"] in all_agg[s]["gene_signed"].index]
            if not vals:
                continue
            mu = float(np.nanmean(vals))
            rows.append({"antigen": str(r["antigen"]).replace("_binder", ""),
                         "gene": r["gene"],
                         "n_splits_in_top": int(r["n_splits_in_top"]),
                         "mean_abs": mu, "sd_abs": float(np.nanstd(vals)),
                         "cv": float(np.nanstd(vals) / max(abs(mu), 1e-12)),
                         "mean_signed": float(np.nanmean(sg)) if sg else np.nan,
                         "tcr_locus": bool(ac.identify_tcr_genes([r["gene"]])[0])})
    summary = (pd.DataFrame(rows, columns=cols) if rows
               else pd.DataFrame(columns=cols))
    if len(summary):
        summary = summary.sort_values(["antigen", "n_splits_in_top", "mean_abs"],
                                      ascending=[True, False, False])
        log("\nmost reproducible genes:")
        log(summary.head(30).round(4).to_string(index=False))
        log("\nTCR-locus among them: {0} of {1}".format(
            int(summary["tcr_locus"].sum()), len(summary)))
    else:
        log("\nNo gene reproduced across splits. Empty summary written.")
    summary.to_csv(os.path.join(OUT, "cross_split_summary.csv"), index=False)

    # ---- prototype comparison ---------------------------------------------
    if cfg["with_cellweighted"]:
        banner("BALANCED vs CELL-WEIGHTED PROTOTYPES")
        rows = []
        for s in splits_done:
            a = all_agg[s]
            if "cw_gene_abs" not in a:
                continue
            for c in a["gene_abs"].index:
                bal = set(a["gene_abs"].loc[c].sort_values(
                    ascending=False).head(args.top_n).index)
                cw = set(a["cw_gene_abs"].loc[c].sort_values(
                    ascending=False).head(args.top_n).index)
                rows.append({"split": s,
                             "antigen": str(c).replace("_binder", ""),
                             "overlap": len(bal & cw) / float(args.top_n)})
        if rows:
            pe = pd.DataFrame(rows).pivot_table(index="antigen", columns="split",
                                                values="overlap")
            pe["mean"] = pe.mean(axis=1)
            pe.to_csv(os.path.join(OUT, "prototype_comparison.csv"))
            log(pe.round(3).to_string())

    # ---- metadata ----------------------------------------------------------
    meta = {
        "baseline": args.baseline,
        "splits": splits_done, "n_steps": args.n_steps,
        "cell_batch": args.cell_batch, "step_chunk": args.step_chunk,
        "top_n": args.top_n, "proto_set": args.proto_set,
        "restrict_set": args.restrict_set,
        "with_cellweighted": cfg["with_cellweighted"],
        "normalisation": ("per split, per class: divide by median |delta| "
                          "(class_median mode) -- direction-preserving, "
                          "threshold-free"),
        "completeness_median_of_medians": float(comp["median_rel_err"].median()),
        "completeness_mean_of_means": float(comp["mean_rel_err"].mean()),
        "frac_delta_negative_overall": overall_neg,
        "max_cross_split_agreement": best,
        "runtime_hours": (time.time() - t_start) / 3600.0,
        "scope": "representation only; no claim about binding mechanism",
    }
    with open(os.path.join(OUT, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    banner("DONE  ({0:.2f} h)".format(meta["runtime_hours"]))
    log("results : {0}".format(OUT))
    log("figures : {0}".format(FIG))
    return 0


if __name__ == "__main__":
    sys.exit(main())
