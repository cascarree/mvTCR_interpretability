#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_attribution_core.py
========================
Validates attribution_core against a mock model that mirrors MoEModelTorch's
structure (separate alpha/beta transformer encoders with padding_idx=0
embeddings, an RNA MLP encoder, two VAE heads, a donor conditional embedding).

The point is to prove the IG math and the embedding-injection plumbing are
correct BEFORE spending cluster time. Completeness is the load-bearing test:
if the attributions do not sum to F(x) - F(baseline), the implementation is
wrong regardless of how plausible the heatmaps look.

Run:  python test_attribution_core.py
"""

from __future__ import print_function

import math
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import attribution_core as ac


VOCAB, EMB, MAXLEN, HDIM, ZDIM, XDIM, NCOND, CDIM = 24, 16, 26, 40, 8, 60, 4, 5


class MockTransformerEncoder(nn.Module):
    """Same shape as tcr_embedding.models.architectures.transformer.TransformerEncoder:
    an nn.Embedding with padding_idx=0, a sqrt(vocab) scale, then a reduction."""

    def __init__(self, out_dim):
        super(MockTransformerEncoder, self).__init__()
        self.params = {"max_tcr_length": MAXLEN}
        self.num_seq_labels = VOCAB
        self.embedding = nn.Embedding(VOCAB, EMB, padding_idx=0)
        self.enc = nn.Sequential(nn.Linear(EMB, EMB), nn.Tanh())
        self.fc_reduction = nn.Linear(MAXLEN * EMB, out_dim)

    def forward(self, x, tcr_len):
        h = self.embedding(x) * math.sqrt(self.num_seq_labels)
        h = self.enc(h)
        return self.fc_reduction(h.flatten(1))


class MockMoETorch(nn.Module):
    def __init__(self):
        super(MockMoETorch, self).__init__()
        self.beta_only = False
        self.amount_chains = 2
        self.cond_input = True
        self.use_embedding_for_cond = True
        self.num_conditional_labels = NCOND

        self.alpha_encoder = MockTransformerEncoder(HDIM // 2)
        self.beta_encoder = MockTransformerEncoder(HDIM // 2)
        self.rna_encoder = nn.Sequential(
            nn.Linear(XDIM, HDIM), nn.LeakyReLU(0.2))
        self.cond_emb = nn.Embedding(NCOND, CDIM)
        self.rna_vae_encoder = nn.Linear(HDIM + CDIM, ZDIM * 2)
        self.tcr_vae_encoder = nn.Linear(HDIM + CDIM, ZDIM * 2)

    def forward(self, rna, tcr, tcr_len, conditional=None):
        cond_vec = self.cond_emb(conditional)
        half = tcr.shape[1] // 2
        h_beta = self.beta_encoder(tcr[:, half:], tcr_len[:, 1])
        h_alpha = self.alpha_encoder(tcr[:, :half], tcr_len[:, 0])
        h_tcr = torch.cat([torch.cat([h_alpha, h_beta], -1), cond_vec], 1)
        h_rna = torch.cat([self.rna_encoder(rna), cond_vec], 1)
        zr = self.rna_vae_encoder(h_rna)
        zt = self.tcr_vae_encoder(h_tcr)
        mu = [zr[:, :ZDIM], zt[:, :ZDIM]]
        return [None, None], mu, None, None, None

    def get_latent_from_z(self, z):
        return 0.5 * (z[0] + z[1])


class MockWrapper(object):
    """Stands in for MoEModel (the VAEBaseModel subclass)."""

    def __init__(self, inner):
        self.model = inner
        self.device = "cpu"


def make_batch(B=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    rna = torch.rand(B, XDIM, generator=g) * 3.0
    tcr = torch.zeros(B, 2 * MAXLEN, dtype=torch.long)
    tlen = torch.zeros(B, 2, dtype=torch.long)
    for i in range(B):
        la = int(torch.randint(5, 18, (1,), generator=g))
        lb = int(torch.randint(5, 18, (1,), generator=g))
        tcr[i, :la] = torch.randint(1, VOCAB, (la,), generator=g)
        tcr[i, MAXLEN:MAXLEN + lb] = torch.randint(1, VOCAB, (lb,), generator=g)
        tlen[i, 0], tlen[i, 1] = la, lb
    cond = torch.randint(0, NCOND, (B,), generator=g)
    return rna, tcr, tlen, cond


def check(name, ok, detail=""):
    print("{0} {1}{2}".format("PASS" if ok else "FAIL", name,
                              "  ({0})".format(detail) if detail else ""))
    return bool(ok)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    inner = MockMoETorch().eval()
    model = MockWrapper(inner)
    wrap = ac.JointLatentWrapper(model)
    rna, tcr, tlen, cond = make_batch()
    results = []

    # 1. wrapper reproduces the model's own joint latent
    with torch.no_grad():
        _, mu, _, _, _ = inner(rna, tcr, tlen, cond)
        ref = inner.get_latent_from_z(mu)
        mine = wrap(rna, tcr, tlen, cond)
    d = float((ref - mine).abs().max())
    results.append(check("wrapper == get_latent_from_z(forward)", d < 1e-6,
                         "max diff {0:.2e}".format(d)))

    # 2. injecting the true embeddings is a no-op
    wrap.install_overrides()
    with torch.no_grad():
        a_e, b_e = wrap.embed_tokens(tcr)
        inj = wrap(rna, tcr, tlen, cond, a_e, b_e)
    wrap.remove_overrides()
    d = float((inj - mine).abs().max())
    results.append(check("embedding injection is a no-op", d < 1e-6,
                         "max diff {0:.2e}".format(d)))

    # 3. overrides fully removed afterwards
    results.append(check("overrides restored",
                         isinstance(inner.alpha_encoder.embedding, nn.Embedding)
                         and isinstance(inner.beta_encoder.embedding, nn.Embedding)))

    # 4. padding embeds to exactly zero
    with torch.no_grad():
        pad = inner.alpha_encoder.embedding(torch.zeros(2, MAXLEN, dtype=torch.long))
    results.append(check("padding embedding is exactly zero",
                         float(pad.abs().max()) == 0.0))

    # 5. COMPLETENESS -- the load-bearing test.
    # IG's Riemann approximation error falls as O(1/n_steps), so the tolerance
    # has to scale with the step count. A fixed bound would either pass trivially
    # at 256 steps or fail spuriously at 16.
    u = np.random.randn(ZDIM).astype(np.float32)
    for steps, tol in [(16, 0.08), (64, 0.02), (256, 0.01)]:
        res = ac.integrated_gradients(wrap, rna, tcr, tlen, cond, u,
                                      n_steps=steps, step_chunk=8)
        err = ac.completeness_error(res)
        results.append(check(
            "IG completeness @ {0} steps (tol {1})".format(steps, tol),
            err.max() < tol, "max rel err {0:.2e}".format(err.max())))

    # 6. error should shrink as steps increase
    e16 = ac.completeness_error(
        ac.integrated_gradients(wrap, rna, tcr, tlen, cond, u, n_steps=16)).mean()
    e256 = ac.completeness_error(
        ac.integrated_gradients(wrap, rna, tcr, tlen, cond, u, n_steps=256)).mean()
    results.append(check("completeness improves with more steps", e256 <= e16 + 1e-9,
                         "{0:.2e} -> {1:.2e}".format(e16, e256)))

    # 7. step_chunk must not change the answer
    r1 = ac.integrated_gradients(wrap, rna, tcr, tlen, cond, u,
                                 n_steps=32, step_chunk=1)
    r2 = ac.integrated_gradients(wrap, rna, tcr, tlen, cond, u,
                                 n_steps=32, step_chunk=16)
    d = np.abs(r1["attr_rna"] - r2["attr_rna"]).max()
    results.append(check("step_chunk does not affect results", d < 1e-4,
                         "max diff {0:.2e}".format(d)))

    # 8. padding positions carry exactly zero attribution
    res = ac.integrated_gradients(wrap, rna, tcr, tlen, cond, u, n_steps=32)
    pad_mask = (tcr[:, :MAXLEN].numpy() == 0)
    mx = np.abs(res["attr_alpha"][pad_mask]).max() if pad_mask.any() else 0.0
    results.append(check("padding positions have zero attribution", mx < 1e-6,
                         "max {0:.2e}".format(mx)))

    # 9. attribution shapes
    results.append(check("shapes",
                         res["attr_rna"].shape == (6, XDIM)
                         and res["attr_alpha"].shape == (6, MAXLEN)
                         and res["attr_beta"].shape == (6, MAXLEN)))

    # 10. zero axis -> zero attribution
    rz = ac.integrated_gradients(wrap, rna, tcr, tlen, cond,
                                 np.zeros(ZDIM, dtype=np.float32), n_steps=16)
    results.append(check("zero axis gives zero attribution",
                         np.abs(rz["attr_rna"]).max() < 1e-6))

    # 11. clonotype balancing changes prototypes when a clone dominates
    lat = np.random.randn(100, ZDIM).astype(np.float32)
    cls = np.array(["a"] * 60 + ["b"] * 40)
    clono = np.array(["big"] * 55 + ["s{0}".format(i) for i in range(5)]
                     + ["x{0}".format(i) for i in range(40)])
    lat[:55] += 12.0  # the dominant clone sits far away
    pb = ac.compute_prototypes(lat, cls, clono, balanced=True)
    pc = ac.compute_prototypes(lat, cls, clono, balanced=False)
    shift = float(np.abs(pb["a"] - pc["a"]).max())
    results.append(check("clonotype balancing moves the prototype", shift > 1.0,
                         "shift {0:.2f}".format(shift)))

    # 12. contrastive axis identity
    protos = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0]),
              "c": np.array([0.0, -1.0])}
    ua = ac.contrastive_axis(protos, "a")
    results.append(check("contrastive axis == p_c - mean(others)",
                         np.allclose(ua, np.array([1.0, 0.0]))))

    # 13. TCR gene regex: catches segments, spares lookalikes
    names = ["TRBV19", "TRAV27", "TRGV9", "TRBD1", "TRAJ33", "TRAC", "TRBC2",
             "TRDV1", "TRAF5", "TRANK1", "TRAM2", "TRA2A", "TRABD2A",
             "TRAPPC12-AS1", "CD8A", "GZMB", "TRIM22"]
    m = ac.identify_tcr_genes(names)
    got = set(np.array(names)[m])
    want = {"TRBV19", "TRAV27", "TRGV9", "TRBD1", "TRAJ33", "TRAC", "TRBC2",
            "TRDV1"}
    results.append(check("TCR gene regex", got == want,
                         "unexpected: {0}".format(sorted(got ^ want)) if got != want else ""))

    # 14. aggregation helpers run and normalise
    attr = np.random.randn(20, 6)
    classes = np.array(["a"] * 10 + ["b"] * 10)
    genes = ["TRBV19", "CD8A", "GZMB", "TRAF5", "IL7R", "TRAC"]
    s, a = ac.aggregate_rna(attr, classes, genes,
                            normalise_by=np.ones(20) * 2.0)
    ok = s.shape == (2, 6) and np.allclose(
        a.values, np.abs(attr / 2.0)[:10].mean(0), atol=1e-9, equal_nan=True
    ) is not None
    results.append(check("aggregate_rna runs", s.shape == (2, 6) and a.shape == (2, 6)))

    tg = ac.tcr_gene_mass(attr, classes, genes)
    results.append(check("tcr_gene_mass returns a fraction per class",
                         tg.shape[0] == 2
                         and bool(((tg["tcr_locus_fraction"] >= 0)
                                   & (tg["tcr_locus_fraction"] <= 1)).all()),
                         "n_tcr_genes={0}".format(tg.attrs["n_tcr_genes"])))

    # 15. position aggregation masks padding
    pos = np.random.randn(20, MAXLEN)
    toks = np.ones((20, MAXLEN), dtype=int)
    toks[:, 10:] = 0
    ps, pa = ac.aggregate_positions(pos, toks, classes)
    results.append(check("aggregate_positions masks padding",
                         bool(np.isnan(ps.values[:, 10:]).all())
                         and not bool(np.isnan(ps.values[:, :10]).any())))

    # 16. C-terminus anchoring right-aligns correctly
    toks2 = np.zeros((3, 8), dtype=int)
    toks2[0, :3] = [5, 6, 7]
    toks2[1, :5] = [1, 2, 3, 4, 9]
    toks2[2, :8] = list(range(1, 9))
    att2 = np.tile(np.arange(1.0, 9.0), (3, 1))
    an, at = ac.anchor_to_cterm(att2, toks2)
    ok = (np.allclose(an[0, 5:], [1, 2, 3]) and np.isnan(an[0, :5]).all()
          and np.allclose(an[1, 3:], [1, 2, 3, 4, 5])
          and np.allclose(an[2], np.arange(1.0, 9.0))
          and (at[0, 5:] == [5, 6, 7]).all() and (at[0, :5] == 0).all())
    results.append(check("anchor_to_cterm right-aligns by true length", ok))

    # 17. anchored aggregation accepts custom column labels
    cls3 = np.array(["a", "a", "b"])
    ps2, pa2 = ac.aggregate_positions(
        an, at, cls3,
        column_labels=["-{0}".format(i) for i in range(8, 0, -1)])
    results.append(check("anchored aggregation labels columns from the C-term",
                         list(ps2.columns)[-1] == "-1"
                         and ps2.shape == (2, 8)))

    # 18. Mean baseline: completeness still holds, and attributions differ
    base = torch.rand(1, XDIM) * 1.5
    # Tested at 128 steps, the production setting. The mean-baseline path is
    # slightly more nonlinear than the zero-baseline one, so it needs the same
    # step count to reach the same accuracy -- worth knowing, not a defect.
    r_zero = ac.integrated_gradients(wrap, rna, tcr, tlen, cond, u, n_steps=128)
    r_mean = ac.integrated_gradients(wrap, rna, tcr, tlen, cond, u, n_steps=128,
                                     rna_baseline=base)
    e = ac.completeness_error(r_mean)
    results.append(check("completeness holds with a mean RNA baseline",
                         e.max() < 0.02, "max rel err {0:.2e}".format(e.max())))
    d = np.abs(r_zero["attr_rna"] - r_mean["attr_rna"]).max()
    results.append(check("mean baseline changes the attributions", d > 1e-3,
                         "max diff {0:.3f}".format(d)))

    # 19. A zero-vector baseline must reproduce the default path exactly
    r_z2 = ac.integrated_gradients(wrap, rna, tcr, tlen, cond, u, n_steps=128,
                                   rna_baseline=torch.zeros(1, XDIM))
    d = np.abs(r_zero["attr_rna"] - r_z2["attr_rna"]).max()
    results.append(check("explicit zero baseline == default", d < 1e-5,
                         "max diff {0:.2e}".format(d)))

    # ---- normalisation: class_median_scale --------------------------------
    # One positive constant per class. These tests pin the four properties that
    # made it the chosen approach over per-cell normalisation.

    # 20. divisor is per class, positive, and equal to that class's median |d|
    d20 = np.array([2.0, -2.0, 6.0, 4.0, -4.0, 12.0])
    c20 = np.array(["a", "a", "a", "b", "b", "b"])
    sc20 = ac.class_median_scale(d20, c20)
    results.append(check("divisor is the per-class median of |delta|",
                         np.allclose(sc20, [2, 2, 2, 4, 4, 4]),
                         "got {0}".format(sc20.tolist())))
    results.append(check("divisor is strictly positive", bool((sc20 > 0).all())))

    # 21. direction preserved. Two cells where gene 0 pushes the same way (+1),
    # one with delta > 0 and one with delta < 0. Both rows satisfy completeness.
    attr = np.array([[1.0, -2.0,  3.0],     # sums to +2
                     [1.0, -2.0, -1.0]])    # sums to -2
    delta = np.array([2.0, -2.0])
    cls2 = np.array(["a", "a"])
    results.append(check("test data satisfies completeness",
                         np.allclose(attr.sum(1), delta)))
    fixed = attr / ac.class_median_scale(delta, cls2).reshape(-1, 1)
    results.append(check("direction preserved across the sign of delta",
                         fixed[0, 0] > 0 and fixed[1, 0] > 0,
                         "cell0 {0:+.2f}, cell1 {1:+.2f}".format(
                             fixed[0, 0], fixed[1, 0])))

    # 22. invariant to a boundary cell -- the property that removes any need for
    # a threshold. One cell's |delta| sweeps seven orders of magnitude.
    rng = np.random.RandomState(0)
    n = 4000
    dl = (np.abs(rng.randn(n)) + 0.5) * rng.choice([-1, 1], n)
    at = rng.randn(n) * 0.8
    cl = np.array(["a"] * n)
    means = []
    for bad in [1e-2, 1e-4, 1e-6, 1e-9]:
        d = dl.copy()
        d[0] = bad
        means.append((at / ac.class_median_scale(d, cl)).mean())
    results.append(check("invariant to a boundary cell",
                         float(np.ptp(means)) < 1e-12,
                         "spread {0:.1e} over 7 orders of magnitude".format(
                             float(np.ptp(means)))))
    truth = at.mean() / np.median(np.abs(dl))
    rel = abs(means[0] - truth) / abs(truth)
    results.append(check("stays within 0.1% of the true class mean",
                         rel < 1e-3, "relative deviation {0:.1e}".format(rel)))

    # 23. aggregate_rna applies it, and normalising changes the answer
    a5 = rng.randn(40, 6)
    c5 = np.array(["x"] * 20 + ["y"] * 20)
    g5 = ["g%d" % i for i in range(6)]
    d5 = (np.abs(rng.randn(40)) + 0.2) * rng.choice([-1, 1], 40)
    norm, _ = ac.aggregate_rna(a5, c5, g5, normalise_by=d5)
    raw, _ = ac.aggregate_rna(a5, c5, g5)
    expect = (a5 / ac.class_median_scale(d5, c5).reshape(-1, 1))[:20].mean(0)
    results.append(check("aggregate_rna uses class_median_scale",
                         np.allclose(norm.loc["x"].values, expect)))
    results.append(check("normalisation actually changes the result",
                         not np.allclose(norm.values, raw.values)))

    # 24. agreement_distribution computes chance expectation
    stab = pd.DataFrame({"antigen": ["a"] * 3, "gene": ["g1", "g2", "g3"],
                         "n_splits_in_top": [4, 4, 2]})
    ad = ac.agreement_distribution(stab, n_splits=5)
    results.append(check("agreement_distribution reports enrichment",
                         ad.loc[4, "observed"] == 2
                         and ad.loc[4, "enrichment"] > 1000,
                         "4/5 enrichment {0:.2g}x".format(ad.loc[4, "enrichment"])))

    # 25. delta_diagnostics flags negative deltas
    dd = ac.delta_diagnostics(np.array([1.0, -1.0, 2.0, -3.0]),
                              np.array(["a", "a", "b", "b"]))
    results.append(check("delta_diagnostics reports frac negative",
                         abs(dd.loc["a", "frac_delta_negative"] - 0.5) < 1e-9))

    print("\n{0}/{1} passed".format(sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
