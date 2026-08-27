#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
attribution_core.py
===================
Core routines for the Integrated Gradients (IG) interpretability analysis of the
mvTCR joint latent space.

Everything here is architecture-agnostic: dimensions are read off the loaded
model, never hardcoded, because the five pooled splits were tuned separately by
Optuna and do not share hyperparameters.

Design decisions, all verified against the live model by 01_probe_model.py:

  * Joint latent = 0.5 * (mu_rna + mu_tcr).  MoEModelTorch.forward returns
    mu = [mu_rna, mu_tcr] and get_latent_from_z averages them.
  * The encoder-only path reproduces the full forward to 0.000e+00 and matches
    model.get_latent() exactly, so IG skips both transformer decoders.
  * alpha_encoder.embedding and beta_encoder.embedding are separate nn.Embedding
    modules, each called exactly once per forward, so a plain attribute swap is
    enough to inject interpolated embeddings (no call queue needed).
  * padding_idx=0 and aa_to_id['_'] == 0, so the embedding of an all-padding
    sequence is exactly the zero tensor: the TCR baseline is torch.zeros.
  * adata.X is log1p-normalised, so the zero vector is a valid RNA baseline in
    the same space the model was trained on.

Target function
---------------
For antigen class c with prototype p_c and the mean of the other classes p_bar,
the negative squared distance contrast expands as

    F_c(x)                  = -||z||^2 + 2 z.p_c    - ||p_c||^2
    mean_{d!=c} F_d(x)      = -||z||^2 + 2 z.p_bar  - mean||p_d||^2
    F_{c,not c}(x)          = 2 z(x) . (p_c - p_bar) + const

The -||z||^2 terms cancel exactly and the constant drops out of IG (only
F(x) - F(baseline) matters). So the contrastive target is a *linear* projection
of z onto the axis u_c = p_c - p_bar: cheaper and better conditioned than the
plain quadratic F_c.

The donor conditional is held at each cell's true donor and is never
interpolated: the question is what this cell's RNA and CDR3s contribute, not
what its donor contributes.
"""

from __future__ import print_function

import warnings
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# Matches TCR locus V/D/J/C gene segments only.
# Deliberately NOT `startswith('TRA')`, which would also swallow TRAF5, TRANK1,
# TRAM2, TRA2A, TRABD2A and TRAPPC12-AS1 -- real genes with nothing to do with
# the receptor. The repo's own determine_marker_genes makes exactly that mistake.
TCR_GENE_PATTERN = re.compile(r"^TR[ABGD][VDJC]")


# ==============================================================================
# data
# ==============================================================================
def load_split(split, donor="None", verbose=True):
    """Reproduce evaluation/Evaluation_Fig_1.ipynb::load_10x_data exactly.

    Order matters and is preserved:
      1. load the full 128,502-cell object
      2. fit the donor OneHotEncoder on ALL cells (so obsm['donor'] always has
         four columns in donor_1..donor_4 order, matching cond_emb's rows)
      3. filter to const.HIGH_COUNT_ANTIGENS -- note this DROPS 'no_data',
         leaving 61,227 cells across 8 antigen classes
      4. clonotype-grouped split with random_seed = split
    """
    from sklearn.preprocessing import OneHotEncoder
    import tcr_embedding.utils_training as utils
    from tcr_embedding.utils_preprocessing import group_shuffle_split
    import config.constants_10x as const

    adata = utils.load_data("10x")
    if str(donor) != "None":
        adata = adata[adata.obs["donor"] == "donor_{0}".format(donor)]
    else:
        enc = OneHotEncoder(sparse=False)
        arr = adata.obs["donor"].to_numpy().reshape(-1, 1)
        enc.fit(arr)
        adata.obsm["donor"] = enc.transform(arr)

    adata = adata[adata.obs["binding_name"].isin(const.HIGH_COUNT_ANTIGENS)]

    if split != "full":
        seed = int(split)
        train_val, test = group_shuffle_split(
            adata, group_col="clonotype", val_split=0.20, random_seed=seed)
        train, val = group_shuffle_split(
            train_val, group_col="clonotype", val_split=0.25, random_seed=seed)
        adata.obs["set"] = None
        adata.obs.loc[train.obs.index, "set"] = "train"
        adata.obs.loc[val.obs.index, "set"] = "val"
        adata.obs.loc[test.obs.index, "set"] = "test"
        adata = adata[adata.obs["set"].isin(["train", "val", "test"])]

    adata = adata.copy()
    if verbose:
        print("split {0}: {1} cells, {2} antigen classes".format(
            split, adata.shape[0], adata.obs["binding_name"].nunique()))
        print(adata.obs["set"].value_counts().to_dict())
    return adata


def load_model_for_split(adata, split, donor="None", model_name="moe"):
    """Load the checkpoint through the library's own loader."""
    import tcr_embedding.utils_training as utils
    path = ("../saved_models/journal_2/10x/splits/{0}/"
            "10x_donor_{1}_split_{2}_{0}.pt".format(model_name, donor, split))
    return utils.load_model(adata, path)


def identify_tcr_genes(var_names):
    """Boolean mask over var_names marking TCR locus V/D/J/C segments."""
    return np.array([bool(TCR_GENE_PATTERN.match(str(g))) for g in var_names])


# ==============================================================================
# differentiable wrapper
# ==============================================================================
class _EmbeddingOverride(nn.Module):
    """Transparent stand-in for an nn.Embedding.

    With `inject` unset it delegates to the original module, so the model
    behaves normally. With `inject` set it returns that tensor instead, which is
    how IG feeds interpolated embeddings through an encoder that otherwise only
    accepts integer tokens.

    The scaling in TransformerEncoder.forward (`* sqrt(num_seq_labels)`) happens
    downstream of this call, so injected values pass through it exactly as real
    lookups do.
    """

    def __init__(self, orig):
        super(_EmbeddingOverride, self).__init__()
        self.orig = orig
        self.inject = None

    def forward(self, tokens):
        if self.inject is not None:
            return self.inject
        return self.orig(tokens)


class JointLatentWrapper(nn.Module):
    """Differentiable joint-latent-mean wrapper around MoEModelTorch.

    Reimplements only the encoder half of MoEModelTorch.forward. Verified
    numerically identical (0.000e+00) to both the full forward and to
    model.get_latent(..., return_mean=True).
    """

    def __init__(self, mvtcr_model):
        super(JointLatentWrapper, self).__init__()
        self.wrapper = mvtcr_model
        self.inner = mvtcr_model.model
        self.device = mvtcr_model.device
        self.inner.to(self.device)
        self.inner.eval()

        self.beta_only = getattr(self.inner, "beta_only", False)
        self.amount_chains = getattr(self.inner, "amount_chains", 2)
        self.cond_input = getattr(self.inner, "cond_input", False)
        self.use_embedding_for_cond = getattr(
            self.inner, "use_embedding_for_cond", True)
        self.num_conditional_labels = getattr(
            self.inner, "num_conditional_labels", 0)

        self.alpha_emb = None if self.beta_only else self.inner.alpha_encoder.embedding
        self.beta_emb = self.inner.beta_encoder.embedding
        self.emb_dim = self.beta_emb.embedding_dim
        self.max_len = self.inner.beta_encoder.params["max_tcr_length"]
        self._overrides_installed = False

    # -- embedding injection ---------------------------------------------------
    def install_overrides(self):
        if self._overrides_installed:
            return
        self._a_orig = self.alpha_emb
        self._b_orig = self.beta_emb
        if not self.beta_only:
            self._a_wrap = _EmbeddingOverride(self._a_orig)
            self.inner.alpha_encoder.embedding = self._a_wrap
        self._b_wrap = _EmbeddingOverride(self._b_orig)
        self.inner.beta_encoder.embedding = self._b_wrap
        self._overrides_installed = True

    def remove_overrides(self):
        if not self._overrides_installed:
            return
        if not self.beta_only:
            self.inner.alpha_encoder.embedding = self._a_orig
        self.inner.beta_encoder.embedding = self._b_orig
        self._overrides_installed = False

    def _set_inject(self, a_emb, b_emb):
        if not self.beta_only:
            self._a_wrap.inject = a_emb
        self._b_wrap.inject = b_emb

    def _clear_inject(self):
        if not self.beta_only:
            self._a_wrap.inject = None
        self._b_wrap.inject = None

    def embed_tokens(self, tcr):
        """Token ids -> (alpha_emb, beta_emb), using the ORIGINAL lookup."""
        half = tcr.shape[1] // 2
        a_orig = self._a_orig if self._overrides_installed else self.alpha_emb
        b_orig = self._b_orig if self._overrides_installed else self.beta_emb
        if self.beta_only:
            return None, b_orig(tcr)
        return a_orig(tcr[:, :half]), b_orig(tcr[:, half:])

    # -- forward ---------------------------------------------------------------
    def forward(self, rna, tcr, tcr_len, cond, a_emb=None, b_emb=None):
        """Joint latent mean, shape [B, zdim].

        If a_emb / b_emb are given they replace the embedding lookup, which is
        what makes IG over the amino-acid embedding space possible.
        """
        inner = self.inner
        inject = (a_emb is not None) or (b_emb is not None)
        if inject:
            self._set_inject(a_emb, b_emb)
        try:
            cond_vec = None
            if cond is not None:
                if self.use_embedding_for_cond:
                    cond_vec = inner.cond_emb(cond)
                else:
                    cond_vec = torch.nn.functional.one_hot(
                        cond, self.num_conditional_labels).float()

            half = tcr.shape[1] // 2
            beta_seq = tcr[:, half:] if self.amount_chains == 2 else tcr
            beta_len = tcr_len[:, self.amount_chains - 1]
            h_beta = inner.beta_encoder(beta_seq, beta_len)

            if not self.beta_only:
                h_alpha = inner.alpha_encoder(tcr[:, :half], tcr_len[:, 0])
                h_tcr = torch.cat([h_alpha, h_beta], dim=-1)
            else:
                h_tcr = h_beta

            if cond_vec is not None and self.cond_input:
                h_tcr = torch.cat([h_tcr, cond_vec], dim=1)

            h_rna = inner.rna_encoder(rna)
            if cond_vec is not None and self.cond_input:
                h_rna = torch.cat([h_rna, cond_vec], dim=1)

            z_rna = inner.rna_vae_encoder(h_rna)
            mu_rna = z_rna[:, :z_rna.shape[1] // 2]
            z_tcr = inner.tcr_vae_encoder(h_tcr)
            mu_tcr = z_tcr[:, :z_tcr.shape[1] // 2]
            return 0.5 * (mu_rna + mu_tcr)
        finally:
            if inject:
                self._clear_inject()


# ==============================================================================
# batching
# ==============================================================================
def iter_batches(adata, batch_size, conditional="donor", beta_only=False):
    """Yield (rna, tcr, tcr_len, cond, idx) tensors straight from AnnData.

    Mirrors DataLoader.create_datasets: tcr is [alpha_seq | beta_seq]
    concatenated on axis 1, lengths are column-stacked [alpha_len, beta_len],
    and the conditional is argmax over the one-hot, matching JointDataset.
    """
    from scipy import sparse

    n = adata.shape[0]
    if beta_only:
        tcr_all = np.asarray(adata.obsm["beta_seq"])
        len_all = np.vstack([adata.obs["beta_len"]]).T
    else:
        tcr_all = np.concatenate(
            [adata.obsm["alpha_seq"], adata.obsm["beta_seq"]], axis=1)
        len_all = np.vstack([adata.obs["alpha_len"], adata.obs["beta_len"]]).T

    cond_all = None
    if conditional is not None:
        cond_all = np.asarray(adata.obsm[conditional]).argmax(1)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x = adata.X[start:stop]
        if sparse.issparse(x):
            x = x.toarray()
        rna = torch.FloatTensor(np.asarray(x))
        tcr = torch.LongTensor(tcr_all[start:stop])
        tlen = torch.LongTensor(len_all[start:stop])
        cond = (torch.LongTensor(cond_all[start:stop])
                if cond_all is not None else None)
        yield rna, tcr, tlen, cond, np.arange(start, stop)


@torch.no_grad()
def compute_latents(wrapper, adata, batch_size=512, conditional="donor"):
    """Joint latent means for every cell, shape [N, zdim]."""
    dev = wrapper.device
    out = []
    for rna, tcr, tlen, cond, _ in iter_batches(
            adata, batch_size, conditional, wrapper.beta_only):
        z = wrapper(rna.to(dev), tcr.to(dev), tlen.to(dev),
                    None if cond is None else cond.to(dev))
        out.append(z.detach().cpu().numpy())
    return np.vstack(out)


# ==============================================================================
# prototypes
# ==============================================================================
def compute_prototypes(latents, classes, clonotypes=None, balanced=True):
    """Per-class prototypes in latent space.

    balanced=True averages within clonotype first, then across clonotypes, so a
    single expanded clone cannot define its class. This matters here: three of
    the eight antigen classes have their largest clonotype holding 31-55% of
    all their cells.

    Returns {class_label: np.ndarray[zdim]}.
    """
    protos = {}
    for c in np.unique(classes):
        mask = classes == c
        if mask.sum() == 0:
            continue
        z = latents[mask]
        if balanced and clonotypes is not None:
            cl = np.asarray(clonotypes)[mask]
            means = [z[cl == u].mean(axis=0) for u in np.unique(cl)]
            protos[c] = np.mean(means, axis=0)
        else:
            protos[c] = z.mean(axis=0)
    return protos


def contrastive_axis(protos, c):
    """u_c = p_c - mean_{d != c} p_d.

    The target 2 * z.u_c is the linear form the contrastive negative-squared-
    distance objective collapses to, once the -||z||^2 terms cancel.
    """
    others = [v for k, v in protos.items() if k != c]
    if not others:
        raise ValueError("need at least two classes for a contrast")
    return protos[c] - np.mean(others, axis=0)


# ==============================================================================
# integrated gradients
# ==============================================================================
def integrated_gradients(wrapper, rna, tcr, tcr_len, cond, u,
                         n_steps=32, step_chunk=8, target_scale=2.0,
                         rna_baseline=None):
    """IG attributions for one batch of cells against one contrastive axis.

    rna_baseline
        None  -> the zero vector. Defensible in principle (X is log1p-normalised
                 so zero means "no expression"), but pathological in practice:
                 attribution is (x - baseline) * grad, so with baseline 0 the
                 first factor is just x, and the highest-expressed genes
                 (mitochondrial, ribosomal) dominate mechanically. It also puts
                 the baseline off-manifold, which made F(baseline) exceed F(x)
                 for 30% of cells in the first run.
        tensor [1, G] or [G] -> subtracted instead. The mean expression vector
                 turns the question into "what makes this cell unusual" rather
                 than "what makes this cell non-empty".

    The TCR baseline is always the all-padding sequence, which embeds to exactly
    zero because padding_idx=0. That is a genuine absence-of-sequence reference
    and needs no equivalent adjustment.

    The donor conditional is held fixed at each cell's true value.

    Uses the midpoint Riemann rule, which converges faster than left/right
    endpoints for the same step count.

    Returns dict with per-cell:
        attr_rna    [B, G]
        attr_alpha  [B, L]   summed over the embedding dimension
        attr_beta   [B, L]
        delta       [B]      F(x) - F(baseline), the completeness target
        attr_total  [B]      sum of all attributions; should equal delta
    """
    dev = wrapper.device
    rna = rna.to(dev)
    tcr = tcr.to(dev)
    tcr_len = tcr_len.to(dev)
    cond = None if cond is None else cond.to(dev)
    u_t = torch.as_tensor(u, dtype=torch.float32, device=dev).detach()

    B = rna.shape[0]
    wrapper.install_overrides()
    try:
        with torch.no_grad():
            a_emb, b_emb = wrapper.embed_tokens(tcr)
            a_emb = None if a_emb is None else a_emb.detach()
            b_emb = b_emb.detach()

            if rna_baseline is None:
                base_rna = torch.zeros_like(rna)
            else:
                base_rna = torch.as_tensor(
                    rna_baseline, dtype=rna.dtype, device=dev).reshape(1, -1)
                base_rna = base_rna.expand_as(rna).contiguous()

            z_x = wrapper(rna, tcr, tcr_len, cond, a_emb, b_emb)
            zero_a = None if a_emb is None else torch.zeros_like(a_emb)
            zero_b = torch.zeros_like(b_emb)
            z_base = wrapper(base_rna, tcr, tcr_len, cond, zero_a, zero_b)
            delta = target_scale * ((z_x - z_base) @ u_t)

        grad_rna = torch.zeros_like(rna)
        grad_a = None if a_emb is None else torch.zeros_like(a_emb)
        grad_b = torch.zeros_like(b_emb)

        alphas = (torch.arange(n_steps, dtype=torch.float32, device=dev) + 0.5)
        alphas = alphas / float(n_steps)

        for start in range(0, n_steps, step_chunk):
            chunk = alphas[start:start + step_chunk]
            k = chunk.shape[0]

            rna_r = rna.repeat(k, 1)
            base_r = base_rna.repeat(k, 1)
            tcr_r = tcr.repeat(k, 1)
            len_r = tcr_len.repeat(k, 1)
            cond_r = None if cond is None else cond.repeat(k)
            a_scale = chunk.repeat_interleave(B).view(-1, 1)

            # general form: baseline + alpha * (input - baseline).
            # With base 0 this reduces to the previous rna_r * a_scale.
            rna_in = (base_r + a_scale * (rna_r - base_r)
                      ).detach().requires_grad_(True)
            inputs = [rna_in]

            b_r = b_emb.repeat(k, 1, 1)
            b_in = (b_r * a_scale.view(-1, 1, 1)).detach().requires_grad_(True)
            inputs.append(b_in)

            a_in = None
            if a_emb is not None:
                a_r = a_emb.repeat(k, 1, 1)
                a_in = (a_r * a_scale.view(-1, 1, 1)).detach().requires_grad_(True)
                inputs.append(a_in)

            z = wrapper(rna_in, tcr_r, len_r, cond_r, a_in, b_in)
            target = target_scale * (z @ u_t)
            grads = torch.autograd.grad(target.sum(), inputs)

            g_rna = grads[0].view(k, B, -1).sum(0)
            grad_rna += g_rna
            g_b = grads[1].view(k, B, b_emb.shape[1], -1).sum(0)
            grad_b += g_b
            if a_in is not None:
                g_a = grads[2].view(k, B, a_emb.shape[1], -1).sum(0)
                grad_a += g_a

        grad_rna /= float(n_steps)
        grad_b /= float(n_steps)
        if grad_a is not None:
            grad_a /= float(n_steps)

        attr_rna = (rna - base_rna) * grad_rna
        attr_b_full = b_emb * grad_b
        attr_beta = attr_b_full.sum(-1)
        if a_emb is not None:
            attr_a_full = a_emb * grad_a
            attr_alpha = attr_a_full.sum(-1)
        else:
            # beta_only models have no alpha chain. Return a correctly shaped
            # zero block rather than a width-0 array, so callers that allocate
            # [n_cells, max_tcr_length] can assign into it without a shape error.
            attr_alpha = torch.zeros(B, attr_beta.shape[1], device=dev)

        total = attr_rna.sum(1) + attr_beta.sum(1) + attr_alpha.sum(1)

        return {
            "attr_rna": attr_rna.detach().cpu().numpy(),
            "attr_alpha": attr_alpha.detach().cpu().numpy(),
            "attr_beta": attr_beta.detach().cpu().numpy(),
            "delta": delta.detach().cpu().numpy(),
            "attr_total": total.detach().cpu().numpy(),
        }
    finally:
        wrapper.remove_overrides()
        wrapper.inner.zero_grad(set_to_none=True)


def completeness_error(res):
    """Relative completeness error per cell. IG guarantees this is ~0."""
    d = res["delta"]
    t = res["attr_total"]
    denom = np.maximum(np.abs(d), 1e-8)
    return np.abs(t - d) / denom


# ==============================================================================
# aggregation
# ==============================================================================


def class_median_scale(delta, classes):
    """Per-cell divisor: each class's median |F(x) - F(x')|.

    Normalisation exists for ONE reason. Each split is a separately trained VAE
    whose latent space is identified only up to rotation and scale, so raw
    attribution magnitudes are not comparable between splits. Dividing by a
    quantity in the same units cancels them out.

    One positive constant per class, applied to every cell in that class:

      * direction preserved -- a positive divisor cannot flip a sign
      * no threshold, nothing modified, nothing discarded
      * a median cannot be moved by one extreme cell, so no outlier protection
        is needed
      * relative magnitude preserved -- a cell that barely moved from the
        baseline contributes little to the class mean, which is correct

    The first run instead divided each cell by its own SIGNED delta. That
    inverted every attribution for the 30.4% of cells with delta < 0, and let a
    single boundary cell move a class mean from -0.003 to -19,460. Both failures
    come from making the denominator a per-cell random variable that can
    approach zero and change sign; a per-class median is neither.

    Returns scale, shape [n_cells], strictly positive.
    """
    d = np.abs(np.asarray(delta, dtype=np.float64))
    cl = np.asarray(classes)
    scale = np.ones_like(d)
    for c in np.unique(cl):
        m = cl == c
        med = np.median(d[m])
        scale[m] = med if med > 0 else 1.0
    return scale


def agreement_distribution(stab, n_splits, n_genes=5000, top_n=20, n_classes=8):
    """Cross-split agreement counts, with the expectation under random ranking.

    Reporting only the 5/5 count was a mistake: under random ranking the
    expected number of gene-class pairs reaching 5/5 is ~4e-8, so observing
    zero is uninformative. The enrichment against chance at each level is the
    number that actually says whether attributions reproduce.
    """
    from math import comb
    p = float(top_n) / float(n_genes)
    rows = []
    obs = stab.groupby("n_splits_in_top").size() if len(stab) else {}
    for k in range(1, n_splits + 1):
        o = int(obs.get(k, 0)) if len(stab) else 0
        e = n_classes * n_genes * comb(n_splits, k) * p ** k * (1 - p) ** (n_splits - k)
        rows.append({"n_splits_in_top": k, "observed": o,
                     "expected_by_chance": e,
                     "enrichment": (o / e) if e > 0 else np.nan})
    return pd.DataFrame(rows).set_index("n_splits_in_top")

def aggregate_rna(attr_rna, classes, gene_names, normalise_by=None):
    """Per-class mean signed and mean absolute gene attribution.

    normalise_by: optional per-cell divisor (use the completeness delta) to make
    magnitudes comparable across independently trained splits. Latent spaces are
    identified only up to rotation/scale, so raw magnitudes are NOT comparable
    across splits -- but input-space attributions are, once normalised.
    """
    a = np.asarray(attr_rna, dtype=np.float64)
    if normalise_by is not None:
        a = a / class_median_scale(normalise_by, classes).reshape(-1, 1)
    rows_signed, rows_abs, labels = [], [], []
    for c in np.unique(classes):
        m = classes == c
        rows_signed.append(np.nanmean(a[m], axis=0))
        rows_abs.append(np.nanmean(np.abs(a[m]), axis=0))
        labels.append(c)
    df_s = pd.DataFrame(rows_signed, index=labels, columns=gene_names)
    df_a = pd.DataFrame(rows_abs, index=labels, columns=gene_names)
    return df_s, df_a


def anchor_to_cterm(attr_pos, tokens):
    """Right-align each CDR3 so the last column is the final residue.

    Sequences are stored left-aligned with right padding, which makes position p
    mean "the p-th residue from the N-terminal C". That is exact near the start
    of the loop but smears the other end, because CDR3 lengths run from 3 to 26.
    Anchoring from the C-terminus gives the complementary view; reading both is
    the only way to tell a genuine positional effect from an alignment artefact.

    Returns (anchored_attributions, anchored_tokens).
    """
    a = np.asarray(attr_pos, dtype=np.float64)
    tok = np.asarray(tokens)
    n, L = a.shape
    out = np.full((n, L), np.nan)
    out_tok = np.zeros((n, L), dtype=tok.dtype)
    lengths = (tok != 0).sum(1)
    for i in range(n):
        li = int(lengths[i])
        if li > 0:
            out[i, L - li:] = a[i, :li]
            out_tok[i, L - li:] = tok[i, :li]
    return out, out_tok


def aggregate_positions(attr_pos, tokens, classes, normalise_by=None,
                        column_labels=None):
    """Per-class mean attribution per CDR3 position, masked over padding.

    Padding positions carry exactly zero attribution by construction (input and
    baseline are both the zero embedding), so including them would dilute the
    mean by a varying amount per class. Only real residues contribute.
    """
    a = np.asarray(attr_pos, dtype=np.float64)
    mask = np.asarray(tokens) != 0
    if normalise_by is not None:
        a = a / class_median_scale(normalise_by, classes).reshape(-1, 1)
    a = np.where(mask, a, np.nan)
    rows_signed, rows_abs, labels = [], [], []
    for c in np.unique(classes):
        m = classes == c
        # A position can be padding in every cell of a class (nobody's CDR3 is
        # that long), giving an all-NaN column. That is a legitimate empty
        # result, not an error -- np.nanmean warns anyway, so silence it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rows_signed.append(np.nanmean(a[m], axis=0))
            rows_abs.append(np.nanmean(np.abs(a[m]), axis=0))
        labels.append(c)
    cols = column_labels or ["p{0}".format(i + 1) for i in range(a.shape[1])]
    return (pd.DataFrame(rows_signed, index=labels, columns=cols),
            pd.DataFrame(rows_abs, index=labels, columns=cols))


def tcr_gene_mass(attr_rna, classes, gene_names):
    """Share of total absolute RNA attribution carried by TCR locus genes.

    Reported per class. This is the number that pre-empts the obvious critique
    of the RNA story -- that the RNA modality is partly re-encoding receptor
    identity via V-gene transcripts, which the CDR3-only TCR encoder never sees.
    """
    is_tcr = identify_tcr_genes(gene_names)
    a = np.abs(np.asarray(attr_rna, dtype=np.float64))
    rows = []
    for c in np.unique(classes):
        m = classes == c
        tot = a[m].sum()
        frac = a[m][:, is_tcr].sum() / tot if tot > 0 else np.nan
        rows.append({"antigen": c,
                     "tcr_locus_fraction": frac,
                     "n_cells": int(m.sum())})
    out = pd.DataFrame(rows).set_index("antigen")
    out.attrs["n_tcr_genes"] = int(is_tcr.sum())
    return out


def cross_split_stability(per_split_abs, top_n=20):
    """How often each gene lands in a class's top-n across independent splits.

    Note the five mvTCR splits are five independent clonotype-grouped draws
    (random_seed = split), NOT a 5-fold partition, so their test sets overlap.
    Agreement is therefore somewhat easier to achieve than a true partition
    would make it -- worth stating in the methods.
    """
    classes = per_split_abs[0].index
    rows = []
    for c in classes:
        counts = {}
        for df in per_split_abs:
            if c not in df.index:
                continue
            for g in df.loc[c].sort_values(ascending=False).head(top_n).index:
                counts[g] = counts.get(g, 0) + 1
        for g, k in counts.items():
            rows.append({"antigen": c, "gene": g, "n_splits_in_top": k,
                         "n_splits": len(per_split_abs)})
    return pd.DataFrame(rows).sort_values(
        ["antigen", "n_splits_in_top"], ascending=[True, False])

def mean_expression_baseline(adata, mask=None):
    """Mean log-normalised expression vector, the alternative RNA baseline.

    Computed from training cells only, so the baseline never sees held-out data.
    """
    from scipy import sparse
    X = adata.X if mask is None else adata[mask].X
    if sparse.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(X).mean(axis=0).astype(np.float32)


def save_split_attributions(path, res):
    """Persist EVERYTHING needed to re-aggregate without recomputing.

    The first version omitted attr_total, the cell-weighted arrays and the
    clonotypes, so the crash cost work that was already done. Nothing is left
    out now.
    """
    payload = dict(
        rna=res["attr"]["rna"], alpha=res["attr"]["alpha"],
        beta=res["attr"]["beta"], delta=res["attr"]["delta"],
        total=res["attr"]["total"],
        classes=res["classes"], sets=res["sets"],
        clonotypes=np.asarray(res["clonotypes"]).astype(str),
        tokens_alpha=res["tokens_alpha"], tokens_beta=res["tokens_beta"],
        genes=np.array(res["genes"]),
        baseline=np.array([res.get("baseline", "zero")]),
    )
    if res.get("attr_cellweighted_rna") is not None:
        payload["cw_rna"] = res["attr_cellweighted_rna"]
        payload["cw_delta"] = res["delta_cellweighted"]
    np.savez_compressed(path, **payload)


def delta_diagnostics(delta, classes):
    """Per-class delta health. delta < 0 means the baseline projects HIGHER on
    the class axis than the real cell -- a sign the baseline is off-manifold."""
    d = np.asarray(delta, dtype=np.float64)
    rows = []
    for c in np.unique(classes):
        m = classes == c
        med = float(np.median(np.abs(d[m]))) if m.any() else np.nan
        rows.append({
            "antigen": str(c).replace("_binder", ""),
            "n": int(m.sum()),
            "frac_delta_negative": float((d[m] < 0).mean()),
            "median_abs_delta": med,
            "frac_below_1pct_median": float(
                (np.abs(d[m]) < 0.01 * med).mean()) if med > 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("antigen")

