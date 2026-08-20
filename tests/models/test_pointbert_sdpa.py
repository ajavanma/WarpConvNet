# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDPA equivalence for PPAT attention, and the neighbourhood rule as recorded config.

The failure mode both halves guard is the same one: a change that alters what the model
computes without altering anything that would announce it. SDPA's ``attn_mask`` enters
the logits *after* the scale where the reference adds ``pe`` *before* it, so passing
``pe`` instead of ``pe * scale`` runs clean and silently trains a different model; and a
neighbourhood rule that lives in a module global leaves no trace in the checkpoint at all.
"""

import pytest
import torch

from warpconvnet.models.ppat import (
    Attention,
    PointPatchTransformer,
    _ball_mask,
    _select_first_k,
    build_openshape_pointbert,
    parse_neighborhood_rule,
    query_ball_point,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _attention(rel_pe: bool, backend: str, dropout: float = 0.0, seed: int = 0) -> Attention:
    torch.manual_seed(seed)
    return (
        Attention(
            dim=32, heads=4, dim_head=8, dropout=dropout, rel_pe=rel_pe, attn_backend=backend
        )
        .to(DEVICE)
        .eval()
    )


# ---------------------------------------------------------------------------
# SDPA equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel_pe", [False, True])
def test_sdpa_matches_naive(rel_pe):
    """The two backends compute the same function, WITH the positional bias active.

    The ``rel_pe=True`` case is the one that has teeth. With ``rel_pe=False`` the bias is
    zero, and ``qk * scale + 0`` equals ``(qk + 0) * scale`` no matter which of ``pe`` or
    ``pe * scale`` the implementation passes as ``attn_mask`` -- so that case would pass
    against the wrong call and report nothing. ``test_sdpa_wrong_mask_scale_is_detected``
    pins that this parametrisation is actually discriminating.
    """
    torch.manual_seed(1)
    x = torch.randn(2, 9, 32, device=DEVICE)
    # Real centroid deltas, so the conv bias has the magnitude it has in the model rather
    # than a scale at which `pe` and `pe * scale` would be indistinguishable.
    c = torch.randn(2, 3, 9, device=DEVICE)
    delta = c.unsqueeze(-1) - c.unsqueeze(-2)

    naive, sdpa = _attention(rel_pe, "naive"), _attention(rel_pe, "sdpa")
    sdpa.load_state_dict(naive.state_dict())

    with torch.no_grad():
        a, b = naive(x, delta), sdpa(x, delta)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_rel_pe_bias_is_non_trivial():
    """The bias in the equivalence test must actually move the output.

    Guards the guard: a randomly initialised ``pe`` head whose output happened to be
    ~constant across the (i, j) grid would add a per-row constant, which softmax is
    invariant to -- and the equivalence test above would then be vacuous for the same
    reason the ``pe = 0`` case is.
    """
    torch.manual_seed(1)
    x = torch.randn(2, 9, 32, device=DEVICE)
    c = torch.randn(2, 3, 9, device=DEVICE)
    delta = c.unsqueeze(-1) - c.unsqueeze(-2)

    with_pe, without = _attention(True, "naive"), _attention(False, "naive")
    without.load_state_dict(
        {k: v for k, v in with_pe.state_dict().items() if not k.startswith("pe.")}
    )
    with torch.no_grad():
        assert not torch.allclose(with_pe(x, delta), without(x, delta), rtol=1e-3, atol=1e-3)


def test_real_sdpa_branch_passes_the_scaled_mask():
    """Assert on the ``attn_mask`` the SHIPPED code hands to SDPA, not on a local copy.

    A test that hand-writes its own SDPA call proves something about the test, not about
    ``Attention.forward``: it would keep passing while the real branch passed the unscaled
    bias. So this spies on the actual call the module makes.

    The final assertion is what makes it a check rather than a restatement -- it pins that
    ``pe * scale`` and ``pe`` are far apart at this fixture's magnitudes, so the equality
    above is discriminating between them rather than holding trivially.
    """
    torch.manual_seed(1)
    x = torch.randn(2, 9, 32, device=DEVICE)
    c = torch.randn(2, 3, 9, device=DEVICE)
    delta = c.unsqueeze(-1) - c.unsqueeze(-2)

    m = _attention(True, "sdpa")
    real = torch.nn.functional.scaled_dot_product_attention
    seen = {}

    def spy(q, k, v, attn_mask=None, dropout_p=0.0, scale=None, **kw):
        seen["mask"], seen["scale"], seen["dropout_p"] = attn_mask, scale, dropout_p
        return real(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale, **kw)

    torch.nn.functional.scaled_dot_product_attention = spy
    try:
        with torch.no_grad():
            m(x, delta)
            pe = m.pe(delta)
    finally:
        torch.nn.functional.scaled_dot_product_attention = real

    assert seen["scale"] == m.scale
    assert seen["dropout_p"] == 0.0
    torch.testing.assert_close(seen["mask"], pe * m.scale, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(seen["mask"], pe, rtol=1e-3, atol=1e-3)


def test_unscaled_mask_injected_into_the_real_branch_breaks_equivalence():
    """The negative control: put the bug in the SHIPPED path and watch equivalence fail.

    Passing here means the suite would have caught the bug. It is the mutation test for
    ``test_sdpa_matches_naive`` -- an equivalence assertion is only worth its runtime if
    something is known to make it fail, and this is that something. The injection divides
    the mask back down by ``scale``, which is exactly the ``attn_mask=pe`` mistake.
    """
    torch.manual_seed(1)
    x = torch.randn(2, 9, 32, device=DEVICE)
    c = torch.randn(2, 3, 9, device=DEVICE)
    delta = c.unsqueeze(-1) - c.unsqueeze(-2)

    naive, sdpa = _attention(True, "naive"), _attention(True, "sdpa")
    sdpa.load_state_dict(naive.state_dict())
    real = torch.nn.functional.scaled_dot_product_attention

    def buggy(q, k, v, attn_mask=None, dropout_p=0.0, scale=None, **kw):
        if attn_mask is not None and scale:
            attn_mask = attn_mask / scale  # the bug: the bias never gets scaled
        return real(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale, **kw)

    torch.nn.functional.scaled_dot_product_attention = buggy
    try:
        with torch.no_grad():
            good, broken = naive(x, delta), sdpa(x, delta)
    finally:
        torch.nn.functional.scaled_dot_product_attention = real

    assert not torch.allclose(good, broken, rtol=1e-3, atol=1e-3)


def test_sdpa_dropout_only_in_training():
    """``dropout_p`` must be gated on ``self.training``, as ``nn.Dropout`` was implicitly.

    SDPA takes dropout as a plain float, so the module-mode gate the reference got for
    free from ``nn.Dropout`` has to be written out. Forgetting it drops attention weights
    during evaluation.
    """
    torch.manual_seed(2)
    x = torch.randn(2, 9, 32, device=DEVICE)
    delta = torch.zeros(2, 3, 9, 9, device=DEVICE)

    m = _attention(False, "sdpa", dropout=0.5)
    with torch.no_grad():
        m.eval()
        assert torch.equal(m(x, delta), m(x, delta))  # deterministic when not training
        m.train()
        assert not torch.equal(m(x, delta), m(x, delta))  # stochastic when training


def test_ppat_end_to_end_backends_agree():
    """Whole-model equivalence, not just the attention block in isolation."""
    torch.manual_seed(3)
    kwargs = dict(
        dim=32,
        depth=2,
        heads=2,
        mlp_dim=64,
        sa_dim=16,
        patches=8,
        prad=0.4,
        nsamp=16,
        in_dim=6,
        dim_head=8,
        rel_pe=True,
    )
    naive = PointPatchTransformer(**kwargs, attn_backend="naive").to(DEVICE).eval()
    sdpa = PointPatchTransformer(**kwargs, attn_backend="sdpa").to(DEVICE).eval()
    sdpa.load_state_dict(naive.state_dict())

    x = torch.randn(2, 6, 256, device=DEVICE)
    with torch.no_grad():
        torch.testing.assert_close(naive(x), sdpa(x), rtol=1e-4, atol=1e-4)


def test_sdpa_backward_matches_naive():
    """Gradients too -- a training run reads those, not the forward."""
    torch.manual_seed(4)
    x = torch.randn(2, 9, 32, device=DEVICE)
    c = torch.randn(2, 3, 9, device=DEVICE)
    delta = c.unsqueeze(-1) - c.unsqueeze(-2)

    naive, sdpa = _attention(True, "naive"), _attention(True, "sdpa")
    sdpa.load_state_dict(naive.state_dict())
    for m in (naive, sdpa):
        m(x, delta).square().sum().backward()

    for (name, a), (_, b) in zip(naive.named_parameters(), sdpa.named_parameters()):
        torch.testing.assert_close(
            a.grad, b.grad, rtol=1e-4, atol=1e-4, msg=lambda s: f"{name}: {s}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast path is CUDA-only")
def test_sdpa_matches_naive_at_bf16_without_rel_pe():
    """The SHIPPED configuration: bf16 autocast, ``rel_pe=False``, hence a null ``attn_mask``.

    This is the only combination every released checkpoint actually runs, and it is a
    DIFFERENT KERNEL from the one the ``rel_pe=True`` tests exercise: flash rejects a
    non-null ``attn_mask``, so a bias forces the memory-efficient backend. Testing only the
    biased path leaves the code that runs in production covered by nothing.

    Tolerance is loose because bf16 carries ~3 decimal digits; the fp32 equivalence tests
    are what pin the arithmetic. What this pins is that the production path runs, dispatches,
    and lands in the same place.
    """
    torch.manual_seed(8)
    x = torch.randn(2, 9, 32, device=DEVICE)
    delta = torch.zeros(2, 3, 9, 9, device=DEVICE)

    naive, sdpa = _attention(False, "naive"), _attention(False, "sdpa")
    sdpa.load_state_dict(naive.state_dict())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = naive(x, delta), sdpa(x, delta)
    assert a.dtype == b.dtype == torch.bfloat16
    torch.testing.assert_close(a.float(), b.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flash backend is CUDA-only")
def test_shipped_config_can_use_the_flash_kernel():
    """Pin that ``rel_pe=False`` is flash-eligible, which is WHY it needs its own test.

    Forcing the flash backend and getting an answer is the evidence that the two
    ``rel_pe`` settings dispatch differently -- without it, the claim that the test above
    covers a distinct kernel is an assumption about SDPA's internals rather than a
    measurement of them.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.manual_seed(9)
    x = torch.randn(2, 9, 32, device=DEVICE)
    delta = torch.zeros(2, 3, 9, 9, device=DEVICE)
    m = _attention(False, "sdpa")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        default = m(x, delta)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            flash = m(x, delta)
    torch.testing.assert_close(default.float(), flash.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast path is CUDA-only")
def test_sdpa_runs_under_bf16_autocast_with_rel_pe():
    """The bias must reach SDPA at the query's dtype, which autocast is what decides.

    Every real run of this model is under bf16 autocast, and that is the one configuration
    the fp32 equivalence tests do not exercise: SDPA rejects an ``attn_mask`` whose dtype
    disagrees with the query, and ``pe`` is a conv output whose dtype autocast chooses.
    Tolerance is loose because bf16 has ~3 decimal digits -- this test is about the call
    being well-formed and the answer being in the right place, not about precision.
    """
    torch.manual_seed(7)
    x = torch.randn(2, 9, 32, device=DEVICE)
    c = torch.randn(2, 3, 9, device=DEVICE)
    delta = c.unsqueeze(-1) - c.unsqueeze(-2)

    naive, sdpa = _attention(True, "naive"), _attention(True, "sdpa")
    sdpa.load_state_dict(naive.state_dict())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = naive(x, delta), sdpa(x, delta)
    assert a.dtype == b.dtype == torch.bfloat16
    torch.testing.assert_close(a.float(), b.float(), rtol=2e-2, atol=2e-2)


def test_attn_backend_rejects_unknown():
    with pytest.raises(ValueError, match="attn_backend"):
        Attention(dim=32, heads=4, dim_head=8, attn_backend="flash")


# ---------------------------------------------------------------------------
# Neighbourhood rule as recorded config
# ---------------------------------------------------------------------------


def test_default_neighborhood_is_the_exact_ball():
    """The default must be exact, so a checkpoint cannot quietly carry an approximation."""
    model = build_openshape_pointbert("openshape-pointbert-vitg14-rgb", in_dim=6)
    report = model.ppat_config_report
    assert report["neighborhood"] == "cumsum"
    assert report["neighborhood_rule"] == "cumsum"
    assert report["neighborhood_exact"] is True


def test_config_report_carries_the_cell_size():
    """``voxel`` is not identified by its name: the report must carry the cell size."""
    model = build_openshape_pointbert(
        "openshape-pointbert-vitg14-rgb", in_dim=6, neighborhood="voxel:0.1"
    )
    report = model.ppat_config_report
    assert report["neighborhood_rule"] == "voxel"
    assert report["neighborhood_cell_size"] == 0.1
    assert report["neighborhood_exact"] is False


def test_voxel_requires_an_explicit_cell_size():
    """Cell size changes the patch geometry and must not be derived behind a default."""
    with pytest.raises(ValueError, match="explicit cell size"):
        parse_neighborhood_rule("voxel")
    with pytest.raises(ValueError, match="explicit cell size"):
        PointPatchTransformer(
            dim=32,
            depth=1,
            heads=2,
            mlp_dim=64,
            sa_dim=16,
            patches=8,
            prad=0.4,
            nsamp=16,
            in_dim=6,
            dim_head=8,
            neighborhood="voxel",
        )


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("auto", dict(rule="cumsum", exact=True, cell_size=None, radius_scale=None)),
        ("dense", dict(rule="dense", exact=True, cell_size=None, radius_scale=None)),
        ("knn", dict(rule="knn", exact=False, cell_size=None, radius_scale=None)),
        ("random", dict(rule="random", exact=False, cell_size=None, radius_scale=None)),
        ("linf", dict(rule="linf", exact=False, cell_size=None, radius_scale=1.0)),
        ("l1:1.732", dict(rule="l1", exact=False, cell_size=None, radius_scale=1.732)),
        ("voxel:0.05", dict(rule="voxel", exact=False, cell_size=0.05, radius_scale=None)),
    ],
)
def test_parse_neighborhood_rule(spec, expected):
    assert parse_neighborhood_rule(spec) == expected


def test_parse_rejects_unknown_and_stray_parameters():
    with pytest.raises(ValueError, match="Unknown neighbourhood rule"):
        parse_neighborhood_rule("ball")
    with pytest.raises(ValueError, match="takes no parameter"):
        parse_neighborhood_rule("knn:8")


@pytest.mark.parametrize("spec", ["voxel:0", "voxel:-0.1", "voxel:nan", "l1:0", "linf:inf"])
def test_parse_rejects_nonpositive_or_nonfinite_scales(spec):
    with pytest.raises(ValueError, match="finite and positive"):
        parse_neighborhood_rule(spec)


def test_model_neighborhood_reaches_the_ball_query():
    """The recorded rule must be the one that runs -- a report about an unused setting is worse
    than no report, since it reads as evidence."""
    torch.manual_seed(5)
    model = (
        PointPatchTransformer(
            dim=32,
            depth=1,
            heads=2,
            mlp_dim=64,
            sa_dim=16,
            patches=8,
            prad=0.4,
            nsamp=16,
            in_dim=6,
            dim_head=8,
            neighborhood="knn",
        )
        .to(DEVICE)
        .eval()
    )
    assert model.sa.neighborhood == "knn"

    xyz = torch.randn(2, 256, 3, device=DEVICE)
    new_xyz = xyz[:, :8]
    torch.testing.assert_close(
        query_ball_point(0.4, 16, xyz, new_xyz, backend=model.sa.neighborhood),
        query_ball_point(0.4, 16, xyz, new_xyz, backend="knn"),
    )
    with torch.no_grad():
        assert model(torch.randn(2, 6, 256, device=DEVICE)).shape == (2, 32)


def test_report_tracks_a_post_construction_rule_change():
    """Mutating the rule must move the report with it, and must re-validate.

    The sweep sets ``sa.neighborhood`` between scores, so the rule genuinely does change
    after construction. A report snapshotted in ``__init__`` would keep naming the rule the
    model was BUILT with while a different one executed -- an authoritative-looking run
    record that is wrong, which is worse than having none.
    """
    model = build_openshape_pointbert("openshape-pointbert-vitg14-rgb", in_dim=6)
    assert model.ppat_config_report["neighborhood_exact"] is True

    model.ppat.sa.neighborhood = "voxel:0.05"
    report = model.ppat_config_report
    assert report["neighborhood"] == "voxel:0.05"
    assert report["neighborhood_cell_size"] == 0.05
    assert report["neighborhood_exact"] is False
    assert model.ppat.sa.neighborhood_config["rule"] == "voxel"

    # An invalid rule must be rejected at assignment, not deferred to the next forward,
    # and must leave the previous valid state intact rather than half-applied.
    with pytest.raises(ValueError):
        model.ppat.sa.neighborhood = "voxel"
    assert model.ppat_config_report["neighborhood"] == "voxel:0.05"


def _cell_gather_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import warpconvnet._C as _C
        import warpconvnet.ops.cell_gather  # noqa: F401
    except Exception:
        return False
    return hasattr(_C.coords, "cell_gather")


requires_cell_gather = pytest.mark.skipif(
    not _cell_gather_available(), reason="cell-gather CUDA kernels not built"
)


#: A fixture dense enough that groups actually overflow the cap. 8192 points in a
#: half-width-0.5 cube puts ~270 points inside radius 0.2, which is deliberately close to
#: the real corpus's median of 262 against ``nsamp=64``. The obvious fixture -- uniform
#: points in [-1, 1]^3 -- averages ~17 per ball, so it never reaches the cap and every
#: equality assertion over it tests a far weaker property than it appears to.
_FIXTURE = dict(b=4, n=8192, s=128, radius=0.2, nsample=64)


def _dense_fixture(seed: int):
    torch.manual_seed(seed)
    xyz = (torch.rand(_FIXTURE["b"], _FIXTURE["n"], 3, device=DEVICE) - 0.5).contiguous()
    return xyz, xyz[:, : _FIXTURE["s"]].contiguous()


def _assert_not_vacuous(counts, sel, nsample, expect_overflow: bool):
    """Fail if the data is too thin for an equality assertion over it to mean anything.

    An equality check cannot distinguish "0 differences" from "0 comparisons". The
    verification that produced these tests hit exactly that -- a transposed batch layout fed
    it 6 points per shape and everything passed on nothing -- and the tell was a count, so
    the counts are asserted rather than left to be noticed.

    ``counts`` is the number of candidates each query has UNDER ITS OWN RULE, not under the
    L2 ball: the voxel rule's neighbourhood is a cell block, and measuring it with a sphere
    would be checking the wrong thing while looking rigorous.

    ``expect_overflow`` picks which regime is being claimed. Both matter and they exercise
    different code: overflowing the cap is what tests the ordering rule (which of the
    candidates get kept), while short groups are what tests the padding path (repeat the
    rank-0 entry). A test that only ever sees one of them has covered half the function.
    """
    assert int(counts.min()) > 1, "a group has <2 candidates; the comparison is degenerate"
    assert sel.unique().numel() > nsample, "selection is near-constant; equality is vacuous"
    if expect_overflow:
        assert int((counts > nsample).sum()) > 0, "no group overflows the cap; ordering untested"
    else:
        assert int((counts < nsample).sum()) > 0, "no group is short; padding path untested"


@requires_cell_gather
def test_voxel_cuda_is_bit_identical_to_voxel():
    """``voxel_cuda:C`` must select the SAME indices as ``voxel:C``, not merely similar ones.

    A plausible cheaper kernel fills each group in *cell* order rather than point-index
    order; it passes membership and count checks while drawing a capped group from one
    part of the 3x3x3 block. Exact index equality is what separates those implementations.
    """
    xyz, new_xyz = _dense_fixture(11)
    r, k = _FIXTURE["radius"], _FIXTURE["nsample"]
    # 0.1 overflows the cap (3x3x3 of 0.1 cells holds ~220 here) and 0.05 does not (~28),
    # so the pair covers both the ordering rule and the padding path.
    for cell, overflow in ((0.1, True), (0.05, False)):
        ref = query_ball_point(r, k, xyz, new_xyz, backend=f"voxel:{cell}")
        cud = query_ball_point(r, k, xyz, new_xyz, backend=f"voxel_cuda:{cell}")
        counts = _ball_mask("voxel", r, xyz, new_xyz, voxel=cell).sum(-1)
        _assert_not_vacuous(counts, ref, k, expect_overflow=overflow)
        assert torch.equal(ref, cud), f"voxel_cuda:{cell} diverged from voxel:{cell}"


@requires_cell_gather
def test_ball_cuda_matches_the_float64_neighbourhood():
    """``ball_cuda`` must equal the ball computed in float64 -- the arbiter neither path uses.

    Stated against float64 rather than against ``cumsum`` on purpose. ``cumsum`` inherits
    the reference's ``-2ab + |a|^2 + |b|^2`` expansion, which can round differently at the
    radius boundary. Comparing with float64 checks the direct-distance CUDA predicate
    independently of that operation-order difference.
    """
    xyz, new_xyz = _dense_fixture(12)
    radius, nsample = _FIXTURE["radius"], _FIXTURE["nsample"]

    d2 = (new_xyz.double().unsqueeze(2) - xyz.double().unsqueeze(1)).pow(2).sum(-1)
    inside = d2 <= radius**2
    truth = _select_first_k(inside, nsample)

    cud = query_ball_point(radius, nsample, xyz, new_xyz, backend="ball_cuda")
    # Overflow exercises selection at the cap as well as membership.
    _assert_not_vacuous(inside.sum(-1), truth, nsample, expect_overflow=True)
    assert torch.equal(cud, truth)


@requires_cell_gather
def test_cell_gather_backends_are_deterministic():
    """Same input twice, same indices -- a capped 27-way merge is where a race would live."""
    # The dense fixture on purpose: a race in a capped 27-way merge is likeliest when
    # groups are actually filling, which the sparse fixture never makes them do.
    xyz, new_xyz = _dense_fixture(13)
    r, k = _FIXTURE["radius"], _FIXTURE["nsample"]
    for backend in ("ball_cuda", "voxel_cuda:0.1"):
        a = query_ball_point(r, k, xyz, new_xyz, backend=backend)
        b = query_ball_point(r, k, xyz, new_xyz, backend=backend)
        assert torch.equal(a, b), f"{backend} is not deterministic"


def test_cuda_rules_parse_and_are_classified_correctly():
    """Parsing does not need the kernels, so this runs everywhere."""
    assert parse_neighborhood_rule("ball_cuda") == dict(
        rule="ball_cuda", exact=True, cell_size=None, radius_scale=None
    )
    assert parse_neighborhood_rule("voxel_cuda:0.1") == dict(
        rule="voxel_cuda", exact=False, cell_size=0.1, radius_scale=None
    )
    # The cell size is as sharp for the kernel as for the torch path.
    with pytest.raises(ValueError, match="explicit cell size"):
        parse_neighborhood_rule("voxel_cuda")


def test_auto_still_resolves_to_cumsum():
    """The fast exact path must NOT become the default by being added.

    ``ball_cuda`` is 26x faster and matches float64 more closely than ``cumsum`` does, which
    is exactly the argument that would make flipping the default feel like a cleanup. It is
    not one: the rule a checkpoint was trained under is a property of that checkpoint, and
    ``parse_neighborhood_rule`` exists to keep it explicit.
    """
    assert parse_neighborhood_rule("auto")["rule"] == "cumsum"
    model = build_openshape_pointbert("openshape-pointbert-vitg14-rgb", in_dim=6)
    assert model.ppat_config_report["neighborhood_rule"] == "cumsum"


def test_exact_backends_still_agree_after_the_parser_refactor():
    """``cumsum`` / ``dense`` / ``warp`` route through the parser now; they must still match."""
    torch.manual_seed(6)
    xyz = torch.rand(2, 512, 3, device=DEVICE) * 2 - 1
    new_xyz = xyz[:, :16]
    ref = query_ball_point(0.4, 32, xyz, new_xyz, backend="dense")
    torch.testing.assert_close(query_ball_point(0.4, 32, xyz, new_xyz, backend="cumsum"), ref)
    torch.testing.assert_close(query_ball_point(0.4, 32, xyz, new_xyz, backend="auto"), ref)
