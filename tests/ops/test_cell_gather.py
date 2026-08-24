# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the capped cell-neighbourhood gather kernels.

voxel_block_gather (P1) is validated on MEMBERSHIP, never index equality:
its selection rule (first ``nsample`` in cell/index order, no distance test)
is deliberately different from every existing path. capped_ball_query (P2)
implements the reference selection rule (lowest in-radius indices, ascending,
true d^2 <= r^2 predicate) and is compared for full equality against a direct
torch reference of that rule.

Shapes include the corpus's measured worst cases, not just random clouds:
a near-line of extent [0.0001, 0.0003, 1.0099] whose 10,000 points occupy a
handful of cells, and a heavy-tail cloud where nearly all points fall in one
query's neighbourhood.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

# Skip, never fail to import. These tests need a `_C` built from a tree that
# contains cell_gather_kernels.cu, and a checkout or an installed extension
# predating it must not take the whole repo's pytest run down with a collection
# error -- a failing test is information, an uncollectable module is a wall.
try:
    import warpconvnet._C as _C

    from warpconvnet.ops.cell_gather import (
        capped_ball_query,
        cell_nearest_k,
        voxel_block_gather,
    )

    if not hasattr(_C.coords, "cell_gather"):
        raise ImportError("_C.coords.cell_gather missing")
except ImportError as exc:  # pragma: no cover - environment guard
    pytest.skip(
        f"cell gather kernels unavailable ({exc}); rebuild the extension with "
        "`uv pip install -e . --no-build-isolation --no-deps`",
        allow_module_level=True,
    )

DEVICE = "cuda:0"
CELL_NEAREST_AVAILABLE = hasattr(_C.coords, "cell_nearest_k")
requires_cell_nearest = pytest.mark.skipif(
    not CELL_NEAREST_AVAILABLE,
    reason="cell-nearest-k CUDA kernel not built",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _offsets(counts, device=DEVICE):
    off = torch.zeros(len(counts) + 1, dtype=torch.int32, device=device)
    off[1:] = torch.cumsum(torch.tensor(counts, device=device), dim=0)
    return off


def _make_batch(clouds, n_query_each, seed=0):
    """Concatenate per-batch clouds; queries are random subsets of each cloud."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    points = torch.cat(clouds).to(DEVICE)
    queries = []
    for cloud, nq in zip(clouds, n_query_each):
        sel = torch.randperm(cloud.shape[0], generator=g)[:nq]
        queries.append(cloud[sel])
    queries = torch.cat(queries).to(DEVICE)
    ref_off = _offsets([c.shape[0] for c in clouds])
    qry_off = _offsets(list(n_query_each))
    return points, ref_off, queries, qry_off


def _rand_cloud(n, seed, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand(n, 3, generator=g) * 2 - 1) * scale


def _near_line_cloud(n, seed):
    """The corpus's degenerate near-line: extent [0.0001, 0.0003, 1.0099]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.rand(n, 3, generator=g)
    return u * torch.tensor([0.0001, 0.0003, 1.0099]) - torch.tensor([0.00005, 0.00015, 0.505])


def _block_membership(points, ref_off, queries, qry_off, cell_size):
    """(M, N) bool: point j is in query i's 3x3x3 cell block AND same batch."""
    pc = torch.floor(points / cell_size)
    qc = torch.floor(queries / cell_size)
    cheb = (qc.unsqueeze(1) - pc.unsqueeze(0)).abs().max(-1).values  # (M, N)
    B = ref_off.shape[0] - 1
    pb = torch.repeat_interleave(
        torch.arange(B, device=points.device), (ref_off[1:] - ref_off[:-1]).long()
    )
    qb = torch.repeat_interleave(
        torch.arange(B, device=points.device), (qry_off[1:] - qry_off[:-1]).long()
    )
    same_batch = qb.unsqueeze(1) == pb.unsqueeze(0)
    return (cheb <= 1.0) & same_batch


def _sphere_membership(points, ref_off, queries, qry_off, radius):
    """(M, N) bool: true d^2 <= r^2, computed directly, AND same batch."""
    d = queries.unsqueeze(1) - points.unsqueeze(0)
    d2 = (d * d).sum(-1)
    B = ref_off.shape[0] - 1
    pb = torch.repeat_interleave(
        torch.arange(B, device=points.device), (ref_off[1:] - ref_off[:-1]).long()
    )
    qb = torch.repeat_interleave(
        torch.arange(B, device=points.device), (qry_off[1:] - qry_off[:-1]).long()
    )
    same_batch = qb.unsqueeze(1) == pb.unsqueeze(0)
    return (d2 <= radius * radius) & same_batch


def _reference_capped_ball(membership, ref_off, qry_off, nsample):
    """The reference selection rule on a membership matrix: per query, the
    ``nsample`` lowest member indices ascending; pad with the first member;
    batch-local 0 (global ref_off[b]) when empty."""
    M, N = membership.shape
    device = membership.device
    idx = torch.arange(N, device=device).unsqueeze(0).expand(M, N)
    key = torch.where(membership, idx, torch.full_like(idx, N))
    ordered = key.sort(dim=1).values[:, :nsample]  # ascending member indices, N-padded
    counts = membership.sum(1)
    B = qry_off.shape[0] - 1
    qb = torch.repeat_interleave(
        torch.arange(B, device=device), (qry_off[1:] - qry_off[:-1]).long()
    )
    first = torch.where(counts > 0, ordered[:, 0], ref_off[qb].long())
    rank = torch.arange(nsample, device=device).unsqueeze(0)
    return torch.where(rank < counts.unsqueeze(1), ordered, first.unsqueeze(1)).int()


def _assert_p1_contract(out, membership, ref_off, qry_off, nsample):
    """The P1 correctness contract: membership, distinct counts, padding."""
    M = out.shape[0]
    counts = membership.sum(1)
    B = qry_off.shape[0] - 1
    qb = torch.repeat_interleave(
        torch.arange(B, device=out.device), (qry_off[1:] - qry_off[:-1]).long()
    )
    for i in range(M):
        row = out[i].long()
        k = int(min(int(counts[i]), nsample))
        if k == 0:
            assert (row == ref_off[qb[i]].long()).all(), f"empty group fill wrong at query {i}"
            continue
        # every returned index lies in the query's 3x3x3 block
        assert membership[i][row].all(), f"out-of-block index at query {i}"
        # distinct = min(block_count, nsample); unique except padding
        head = row[:k]
        assert head.unique().numel() == k, f"duplicate non-pad index at query {i}"
        assert torch.unique(row).numel() == k, f"distinct count wrong at query {i}"
        # padding repeats the first hit
        assert (row[k:] == head[0]).all(), f"padding is not the first hit at query {i}"


# ---------------------------------------------------------------------------
# Public-boundary validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_points,n_queries,ref_values,query_values,match",
    [
        (1, 1, [-1, 0], [0, 1], "ref_offsets must start at 0"),
        (1, 1, [0, 2], [0, 1], "ref_offsets must end at 1"),
        (1, 1, [0, 2, 1], [0, 1, 1], "ref_offsets must be monotonically"),
        (2, 1, [0, 1, 2], [0, 2, 1], "query_offsets must be monotonically"),
        (1, 1, [0, 1], [0, 0], "query_offsets must end at 1"),
        (1, 1, [0, 1], [0, 1, 1], "same batch count"),
    ],
)
def test_cell_gather_rejects_malformed_offsets_before_launch(
    n_points, n_queries, ref_values, query_values, match
):
    points = torch.zeros(n_points, 3, device=DEVICE)
    queries = torch.zeros(n_queries, 3, device=DEVICE)
    # CPU int64 offsets are supported and canonicalized by the wrapper; the
    # malformed values must be rejected before they can become kernel pointers.
    ref_offsets = torch.tensor(ref_values, dtype=torch.int64)
    query_offsets = torch.tensor(query_values, dtype=torch.int64)
    with pytest.raises(ValueError, match=match):
        voxel_block_gather(points, ref_offsets, queries, query_offsets, 0.1, 4)


@pytest.mark.parametrize("cell_size", [0.0, -0.1, float("inf"), float("nan")])
def test_voxel_block_gather_rejects_nonpositive_or_nonfinite_cell_size(cell_size):
    points = torch.zeros(1, 3, device=DEVICE)
    offsets = _offsets([1])
    with pytest.raises(ValueError, match="cell_size must be finite and positive"):
        voxel_block_gather(points, offsets, points, offsets, cell_size, 4)


@pytest.mark.parametrize("radius", [0.0, -0.1, float("inf"), float("nan")])
def test_capped_ball_query_rejects_nonpositive_or_nonfinite_radius(radius):
    points = torch.zeros(1, 3, device=DEVICE)
    offsets = _offsets([1])
    with pytest.raises(ValueError, match="radius must be finite and positive"):
        capped_ball_query(points, offsets, points, offsets, radius, 4)


def test_cell_gather_rejects_nonpositive_nsample():
    points = torch.zeros(1, 3, device=DEVICE)
    offsets = _offsets([1])
    with pytest.raises(ValueError, match="nsample must be positive"):
        voxel_block_gather(points, offsets, points, offsets, 0.1, 0)


@pytest.mark.parametrize("nsample", [True, 1.5])
def test_cell_gather_rejects_bool_or_noninteger_nsample(nsample):
    points = torch.zeros(1, 3, device=DEVICE)
    offsets = _offsets([1])
    with pytest.raises(TypeError, match="nsample must be an integer"):
        voxel_block_gather(points, offsets, points, offsets, 0.1, nsample)


def test_cell_gather_rejects_bool_geometry_scalars():
    points = torch.zeros(1, 3, device=DEVICE)
    offsets = _offsets([1])
    with pytest.raises(TypeError, match="cell_size must be a real number"):
        voxel_block_gather(points, offsets, points, offsets, True, 4)
    with pytest.raises(TypeError, match="radius must be a real number"):
        capped_ball_query(points, offsets, points, offsets, True, 4)


def test_cell_gather_rejects_nonfloating_points_or_queries():
    floating = torch.zeros(1, 3, device=DEVICE)
    integer = torch.zeros(1, 3, dtype=torch.int32, device=DEVICE)
    offsets = _offsets([1])
    for points, queries in ((integer, floating), (floating, integer)):
        with pytest.raises(TypeError, match="floating-point dtypes"):
            voxel_block_gather(points, offsets, queries, offsets, 0.1, 4)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cell_gather_rejects_mixed_cuda_devices():
    points = torch.zeros(1, 3, device="cuda:0")
    queries = torch.zeros(1, 3, device="cuda:1")
    offsets = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(ValueError, match="same CUDA device"):
        voxel_block_gather(points, offsets, queries, offsets, 0.1, 4)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cell_gather_restores_the_callers_cuda_device():
    original = torch.cuda.current_device()
    target = (original + 1) % torch.cuda.device_count()
    points = torch.zeros(1, 3, device=f"cuda:{target}")
    offsets = torch.tensor([0, 1], dtype=torch.int32)
    voxel_block_gather(points, offsets, points, offsets, 0.1, 1)
    assert torch.cuda.current_device() == original


def test_cell_gather_binding_rejects_cpu_auxiliary_tensor():
    points = torch.zeros(1, 3, device=DEVICE)
    queries = points.clone()
    query_batch = torch.zeros(1, dtype=torch.int32)  # deliberately CPU
    ref_offsets = _offsets([1])
    sorted_order = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    cell_starts = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    cell_counts = torch.ones(1, dtype=torch.int32, device=DEVICE)
    keys = torch.zeros(16, dtype=torch.int64, device=DEVICE)
    values = torch.zeros(16, dtype=torch.int32, device=DEVICE)
    out = torch.empty(1, 1, dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError, match="query_batch must be a CUDA tensor"):
        _C.coords.cell_gather(
            points,
            queries,
            query_batch,
            ref_offsets,
            sorted_order,
            cell_starts,
            cell_counts,
            keys,
            values,
            out,
            1,
            1,
            0.1,
            0.0,
            False,
            16,
        )


@pytest.mark.parametrize("invalid_batch", [-1, 1])
def test_cell_gather_binding_invalid_query_batch_fails_closed(invalid_batch):
    points = torch.zeros(1, 3, device=DEVICE)
    queries = points.clone()
    query_batch = torch.full((1,), invalid_batch, dtype=torch.int32, device=DEVICE)
    ref_offsets = _offsets([1])
    sorted_order = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    cell_starts = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    cell_counts = torch.ones(1, dtype=torch.int32, device=DEVICE)
    keys = torch.zeros(16, dtype=torch.int64, device=DEVICE)
    values = torch.zeros(16, dtype=torch.int32, device=DEVICE)
    out = torch.empty(1, 1, dtype=torch.int32, device=DEVICE)
    _C.coords.cell_gather(
        points,
        queries,
        query_batch,
        ref_offsets,
        sorted_order,
        cell_starts,
        cell_counts,
        keys,
        values,
        out,
        1,
        1,
        0.1,
        0.0,
        False,
        16,
    )
    assert out.tolist() == [[-1]]


# ---------------------------------------------------------------------------
# P1: voxel_block_gather
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", [0.1, 0.35])
def test_voxel_block_gather_membership_random(cell):
    clouds = [_rand_cloud(1500, seed=1), _rand_cloud(700, seed=2), _rand_cloud(60, seed=3)]
    points, ro, queries, qo = _make_batch(clouds, [64, 48, 16])
    nsample = 32
    out = voxel_block_gather(points, ro, queries, qo, cell, nsample)
    membership = _block_membership(points, ro, queries, qo, cell)
    _assert_p1_contract(out, membership, ro, qo, nsample)


def test_voxel_block_gather_batch_isolation():
    # Two batches occupying the SAME region: any cross-batch leak shows up.
    clouds = [_rand_cloud(1000, seed=10), _rand_cloud(1000, seed=11)]
    points, ro, queries, qo = _make_batch(clouds, [128, 128])
    out = voxel_block_gather(points, ro, queries, qo, 0.2, 48).long()
    B = 2
    for b in range(B):
        rows = out[qo[b] : qo[b + 1]]
        assert (rows >= ro[b]).all() and (rows < ro[b + 1]).all(), f"cross-batch index, batch {b}"


def test_voxel_block_gather_degenerate_near_line():
    # 10,000 points in a handful of cells; block counts far above nsample.
    clouds = [_near_line_cloud(10000, seed=4), _rand_cloud(5000, seed=5)]
    points, ro, queries, qo = _make_batch(clouds, [96, 96])
    for cell in (0.02, 0.1):
        nsample = 64
        out = voxel_block_gather(points, ro, queries, qo, cell, nsample)
        membership = _block_membership(points, ro, queries, qo, cell)
        _assert_p1_contract(out, membership, ro, qo, nsample)


def test_voxel_block_gather_empty_group_and_padding():
    # Query far from every point: empty block -> batch-local 0.
    cloud0 = _rand_cloud(500, seed=6)
    cloud1 = _rand_cloud(300, seed=7)
    points = torch.cat([cloud0, cloud1]).to(DEVICE)
    ro = _offsets([500, 300])
    queries = torch.tensor(
        [[50.0, 50.0, 50.0], [0.0, 0.0, 0.0], [50.0, 50.0, 50.0]], device=DEVICE
    )
    qo = _offsets([2, 1])
    out = voxel_block_gather(points, ro, queries, qo, 0.1, 16).long()
    assert (out[0] == 0).all()  # empty group, batch 0 -> global 0
    assert (out[2] == 500).all()  # empty group, batch 1 -> ref_off[1]
    membership = _block_membership(points, ro, queries, qo, 0.1)
    _assert_p1_contract(out.int(), membership, ro, qo, 16)


@pytest.mark.parametrize(
    "query_x,wrapped_point_x",
    [(131071.5, -131071.5), (-131071.5, 131071.5)],
)
def test_voxel_block_gather_does_not_wrap_packed_coordinate_neighbors(query_x, wrapped_point_x):
    # The query occupies a representable boundary cell. Its outward neighbor
    # is outside the signed 18-bit range and must be skipped, not masked onto
    # the opposite boundary where point 1 lives.
    points = torch.tensor([[0.0, 100.0, 100.0], [wrapped_point_x, 0.0, 0.0]], device=DEVICE)
    queries = torch.tensor([[query_x, 0.0, 0.0]], device=DEVICE)
    ref_offsets = _offsets([2])
    query_offsets = _offsets([1])
    membership = _block_membership(points, ref_offsets, queries, query_offsets, 1.0)
    assert not membership.any()

    out = voxel_block_gather(points, ref_offsets, queries, query_offsets, cell_size=1.0, nsample=4)
    assert out.tolist() == [[0, 0, 0, 0]]


@pytest.mark.parametrize("cell", [0.1, 0.05])
def test_voxel_block_gather_matches_python_voxel_backend(cell):
    """Bit-equality against the public Python voxel rule.

    The membership contract above passes for ANY in-block subset, including a
    cell-ordered fill that draws every neighbour from one corner of the 3x3x3
    block -- wrong but plausible, and invisible to a count-based check. The
    public rule uses `_select_first_k`, the ``nsample`` lowest point indices in
    the block, so this test pins that selection exactly.
    """
    from warpconvnet.models.ppat import query_ball_point

    B, N, S, nsample = 3, 1200, 96, 32
    g = torch.Generator(device="cpu").manual_seed(40)
    xyz = ((torch.rand(B, N, 3, generator=g) * 2 - 1) * 0.7).to(DEVICE)
    sel = torch.stack([torch.randperm(N, generator=g)[:S] for _ in range(B)]).to(DEVICE)
    new_xyz = torch.gather(xyz, 1, sel.unsqueeze(-1).expand(B, S, 3))

    # radius is unused by the voxel rule; cell_size is the whole geometry.
    ref = query_ball_point(0.2, nsample, xyz, new_xyz, backend=f"voxel:{cell}")

    ro = _offsets([N] * B)
    qo = _offsets([S] * B)
    out = voxel_block_gather(
        xyz.reshape(B * N, 3).contiguous(),
        ro,
        new_xyz.reshape(B * S, 3).contiguous(),
        qo,
        cell,
        nsample,
    ).long()
    local = (out.view(B, S, nsample) - ro[:-1].long().view(B, 1, 1)).contiguous()
    assert torch.equal(local, ref), (
        f"voxel_block_gather differs from the measured voxel:{cell} rule on "
        f"{(local != ref).any(-1).sum().item()} of {B * S} queries"
    )


def test_voxel_block_gather_spread_not_corner():
    """The selected patch must span the block, not collapse into one corner.

    A cell-ordered fill scores well on every membership check and still
    shrinks the patch's spatial extent. Compare the selected points' extent
    against the full block's to catch that ordering error.
    """
    B, N, S, cell, nsample = 2, 4000, 128, 0.15, 32
    g = torch.Generator(device="cpu").manual_seed(41)
    clouds = [(torch.rand(N, 3, generator=g) * 2 - 1) * 0.7 for _ in range(B)]
    points, ro, queries, qo = _make_batch(clouds, [S] * B, seed=41)

    out = voxel_block_gather(points, ro, queries, qo, cell, nsample).long()
    membership = _block_membership(points, ro, queries, qo, cell)

    sel_ext, blk_ext = [], []
    for i in range(out.shape[0]):
        blk = points[membership[i]]
        if blk.shape[0] <= nsample:  # no choice was made; nothing to test
            continue
        sel = points[out[i].unique()]
        sel_ext.append((sel.amax(0) - sel.amin(0)).mean().item())
        blk_ext.append((blk.amax(0) - blk.amin(0)).mean().item())
    assert len(sel_ext) > 32, "test shape did not produce over-full blocks"
    ratio = torch.tensor(sel_ext).mean() / torch.tensor(blk_ext).mean()
    # A uniform subset keeps essentially the full extent; a corner fill of 32
    # from ~90 block points keeps roughly a third of it.
    assert ratio > 0.85, f"selected patch collapsed to {ratio:.2f} of the block extent"


def test_voxel_block_gather_deterministic():
    clouds = [_near_line_cloud(10000, seed=8), _rand_cloud(10000, seed=9)]
    points, ro, queries, qo = _make_batch(clouds, [384, 384])
    a = voxel_block_gather(points, ro, queries, qo, 0.1, 64)
    b = voxel_block_gather(points, ro, queries, qo, 0.1, 64)
    assert torch.equal(a, b), "voxel_block_gather is not bit-identical across runs"


def test_voxel_block_gather_large_nsample_shrinks_block():
    """A large nsample shrinks warps-per-block; the answer must not change.

    Each warp holds its picks in shared memory, so nsample sets how many warps
    fit in a block. That is a launch-shape change on a path nsample=64 never
    takes, and silently truncating the buffer would corrupt a NEIGHBOURING
    warp's output rather than crash.
    """
    clouds = [_near_line_cloud(10000, seed=50), _rand_cloud(6000, seed=51)]
    points, ro, queries, qo = _make_batch(clouds, [32, 32])
    membership = _block_membership(points, ro, queries, qo, 0.1)
    assert int(membership.sum(1).max()) > 2048, "shape does not over-fill nsample=2048"
    for nsample in (2048, 4096):
        out = voxel_block_gather(points, ro, queries, qo, 0.1, nsample)
        assert out.shape == (64, nsample)
        _assert_p1_contract(out, membership, ro, qo, nsample)


def test_nsample_beyond_shared_memory_raises():
    points = _rand_cloud(500, seed=52).to(DEVICE)
    ro = _offsets([500])
    queries = points[:4].contiguous()
    qo = _offsets([4])
    with pytest.raises(RuntimeError, match="shared memory"):
        voxel_block_gather(points, ro, queries, qo, 0.1, 48 * 1024 // 4 + 1)


# ---------------------------------------------------------------------------
# P2: capped_ball_query
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [0.2, 0.5])
def test_capped_ball_query_matches_reference_rule(radius):
    clouds = [_rand_cloud(2000, seed=20), _rand_cloud(900, seed=21), _rand_cloud(80, seed=22)]
    points, ro, queries, qo = _make_batch(clouds, [128, 64, 16])
    nsample = 64
    out = capped_ball_query(points, ro, queries, qo, radius, nsample)
    membership = _sphere_membership(points, ro, queries, qo, radius)
    ref = _reference_capped_ball(membership, ro, qo, nsample)
    assert torch.equal(out, ref), (
        f"capped_ball_query differs from the reference rule on "
        f"{(out != ref).any(1).sum().item()} of {out.shape[0]} queries"
    )


def test_capped_ball_query_heavy_tail_and_degenerate():
    # Near-line: nearly all 10k points inside one ball (the p95=4105/max=9511
    # regime); the merge must stop after nsample accepted hits.
    clouds = [_near_line_cloud(10000, seed=23), _rand_cloud(4000, seed=24)]
    points, ro, queries, qo = _make_batch(clouds, [64, 64])
    nsample = 64
    out = capped_ball_query(points, ro, queries, qo, 0.2, nsample)
    membership = _sphere_membership(points, ro, queries, qo, 0.2)
    ref = _reference_capped_ball(membership, ro, qo, nsample)
    assert torch.equal(out, ref)
    # Sanity: the degenerate shape really does exercise the heavy tail. A ball of
    # radius 0.2 covers 0.4 of the line's 1.0099 extent, so ~3,960 of its 10,000
    # points -- right at the corpus's measured p95 of 4,105 in-radius points.
    assert int(membership[:64].sum(1).max()) > 3500


def test_capped_ball_query_boundary_predicate():
    # A point at exactly d^2 == r^2 must be included (<=, computed directly).
    # 0.25 is exactly representable in fp32, so d^2 == r^2 == 0.0625 exactly.
    points = torch.tensor([[0.25, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0]], device=DEVICE)
    ro = _offsets([3])
    queries = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
    qo = _offsets([1])
    out = capped_ball_query(points, ro, queries, qo, 0.25, 4)
    assert 0 in out[0].tolist(), "boundary point at d == radius excluded"
    assert out[0].tolist() == [0, 1, 0, 0]


def test_capped_ball_query_empty_group():
    points = _rand_cloud(400, seed=25).to(DEVICE)
    ro = _offsets([400])
    queries = torch.tensor([[30.0, 30.0, 30.0]], device=DEVICE)
    qo = _offsets([1])
    out = capped_ball_query(points, ro, queries, qo, 0.2, 8)
    assert (out == 0).all()


def test_capped_ball_query_deterministic():
    clouds = [_near_line_cloud(10000, seed=26), _rand_cloud(10000, seed=27)]
    points, ro, queries, qo = _make_batch(clouds, [384, 384])
    a = capped_ball_query(points, ro, queries, qo, 0.2, 64)
    b = capped_ball_query(points, ro, queries, qo, 0.2, 64)
    assert torch.equal(a, b), "capped_ball_query is not bit-identical across runs"


# ---------------------------------------------------------------------------
# Cross-check against the pointbert cumsum backend (expansion predicate)
# ---------------------------------------------------------------------------


def test_capped_ball_query_vs_cumsum_backend():
    """The dense/cumsum reference uses the -2ab+|a|^2+|b|^2 expansion, whose
    cancellation admits boundary points at true distance just above r. Any
    disagreement must therefore be confined to queries whose membership
    differs under the two predicates -- everywhere else the selections must
    be identical."""
    from warpconvnet.models.ppat import _query_ball_point_cumsum

    B, N, S, radius, nsample = 4, 2000, 96, 0.2, 64
    g = torch.Generator(device="cpu").manual_seed(30)
    xyz = (torch.rand(B, N, 3, generator=g) * 2 - 1).to(DEVICE)
    sel = torch.stack([torch.randperm(N, generator=g)[:S] for _ in range(B)]).to(DEVICE)
    new_xyz = torch.gather(xyz, 1, sel.unsqueeze(-1).expand(B, S, 3))

    ref = _query_ball_point_cumsum(radius, nsample, xyz, new_xyz)  # (B, S, ns) batch-local

    points = xyz.reshape(B * N, 3).contiguous()
    queries = new_xyz.reshape(B * S, 3).contiguous()
    ro = _offsets([N] * B)
    qo = _offsets([S] * B)
    out = capped_ball_query(points, ro, queries, qo, radius, nsample).long()
    local = (out.view(B, S, nsample) - ro[:-1].long().view(B, 1, 1)).contiguous()

    disagree = (local != ref).any(-1)  # (B, S)
    if disagree.any():
        # Every disagreeing query must trace to a membership difference
        # between the two boundary predicates; anywhere the predicates agree,
        # the selections must be identical.
        from warpconvnet.models.ppat import _square_distance

        d = new_xyz.unsqueeze(2) - xyz.unsqueeze(1)
        direct = (d * d).sum(-1) <= radius * radius
        expansion = _square_distance(new_xyz, xyz) <= radius * radius
        db, ds = torch.nonzero(disagree, as_tuple=True)
        for b, s in zip(db.tolist(), ds.tolist()):
            assert not torch.equal(direct[b, s], expansion[b, s]), (
                f"selection differs at ({b},{s}) with identical membership -- "
                f"a real bug, not the known boundary-cancellation difference"
            )
    frac = disagree.float().mean().item()
    assert frac < 1e-2, f"capped_ball_query disagrees with cumsum on {frac:.2%} of queries"


# ---------------------------------------------------------------------------
# Fused, order-independent cell-list nearest-k
# ---------------------------------------------------------------------------


@requires_cell_nearest
@pytest.mark.parametrize("cell_size", [0.09, 0.15, 0.25])
def test_cell_nearest_k_matches_direct_dense_knn_in_packed_batches(cell_size):
    clouds = [_rand_cloud(1200, seed=60), _near_line_cloud(900, seed=61)]
    points, ro, queries, qo = _make_batch(clouds, [48, 32], seed=62)
    k = 64

    result = cell_nearest_k(
        points,
        ro,
        queries,
        qo,
        k,
        cell_size=cell_size,
        max_shell=8,
    )
    assert (result.status == 0).all()
    assert (result.counts == k).all()

    for batch in range(2):
        query_slice = slice(int(qo[batch]), int(qo[batch + 1]))
        point_slice = slice(int(ro[batch]), int(ro[batch + 1]))
        distance_sq = (
            (queries[query_slice, None, :] - points[None, point_slice, :]).square().sum(-1)
        )
        expected_distance, expected_local = torch.topk(
            distance_sq,
            k,
            dim=-1,
            largest=False,
            sorted=True,
        )
        expected_global = expected_local + ro[batch].long()
        assert torch.equal(result.indices[query_slice].long(), expected_global)
        torch.testing.assert_close(
            result.squared_distances[query_slice],
            expected_distance,
            rtol=1e-6,
            atol=1e-7,
        )


@requires_cell_nearest
def test_cell_nearest_k_status_ties_and_underfill_contract():
    tied = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        device=DEVICE,
    )
    query = torch.zeros(1, 3, device=DEVICE)
    tied_result = cell_nearest_k(
        tied,
        _offsets([4]),
        query,
        _offsets([1]),
        3,
        cell_size=2.0,
        max_shell=1,
    )
    assert tied_result.status.item() == 0
    assert tied_result.indices.tolist() == [[0, 1, 2]]

    # All points lie in the query cell, so the complete batch is scanned. The
    # result is certified but underfilled: bit 2 (value 4), -1, and +inf.
    short = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
        device=DEVICE,
    )
    short_result = cell_nearest_k(
        short,
        _offsets([3]),
        query,
        _offsets([1]),
        4,
        cell_size=1.0,
        max_shell=0,
    )
    assert short_result.status.item() == 4
    assert short_result.counts.item() == 3
    assert short_result.indices[0, -1].item() == -1
    assert torch.isinf(short_result.squared_distances[0, -1])

    # Two valid picks are found, but a distant third point remains unvisited and
    # shell zero cannot prove exactness: bit 1 (value 2), without underfill.
    sparse = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [10.0, 0.0, 0.0]],
        device=DEVICE,
    )
    budget_result = cell_nearest_k(
        sparse,
        _offsets([3]),
        query,
        _offsets([1]),
        2,
        cell_size=1.0,
        max_shell=0,
    )
    assert budget_result.status.item() == 2
    assert budget_result.indices.tolist() == [[0, 1]]


@requires_cell_nearest
def test_cell_nearest_k_is_deterministic_and_stable_for_untied_permutations():
    points = _rand_cloud(777, seed=63).to(DEVICE)
    queries = points[::53].contiguous()
    ro = _offsets([len(points)])
    qo = _offsets([len(queries)])
    kwargs = {"cell_size": 0.13, "max_shell": 6}

    baseline = cell_nearest_k(points, ro, queries, qo, 32, **kwargs)
    repeated = cell_nearest_k(points, ro, queries, qo, 32, **kwargs)
    assert torch.equal(baseline.indices, repeated.indices)
    assert torch.equal(baseline.squared_distances, repeated.squared_distances)
    assert torch.equal(baseline.status, repeated.status)

    generator = torch.Generator(device=DEVICE).manual_seed(64)
    permutation = torch.randperm(len(points), generator=generator, device=DEVICE)
    permuted = cell_nearest_k(points[permutation], ro, queries, qo, 32, **kwargs)
    mapped_to_original = permutation[permuted.indices.long()]
    assert (baseline.status == 0).all() and (permuted.status == 0).all()
    assert torch.equal(baseline.indices.long(), mapped_to_original)


@requires_cell_nearest
def test_cell_nearest_k_uses_row_index_to_break_distance_ties():
    points = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], device=DEVICE)
    query = torch.zeros(1, 3, device=DEVICE)
    offsets = _offsets([2])
    query_offsets = _offsets([1])

    baseline = cell_nearest_k(
        points,
        offsets,
        query,
        query_offsets,
        1,
        cell_size=1.0,
        max_shell=2,
    )
    permutation = torch.tensor([1, 0], device=DEVICE)
    permuted = cell_nearest_k(
        points[permutation],
        offsets,
        query,
        query_offsets,
        1,
        cell_size=1.0,
        max_shell=2,
    )

    assert baseline.status.item() == 0
    assert permuted.status.item() == 0
    assert baseline.indices.item() == 0
    assert permutation[permuted.indices.long()].item() == 1


@requires_cell_nearest
def test_cell_nearest_k_reports_packed_coordinate_clipping():
    coordinate_max = 131071.0
    points = torch.tensor(
        [[coordinate_max, 0.0, 0.0], [coordinate_max - 0.1, 0.0, 0.0]],
        device=DEVICE,
    )
    result = cell_nearest_k(
        points,
        _offsets([2]),
        points[:1].contiguous(),
        _offsets([1]),
        1,
        cell_size=1.0,
        max_shell=1,
    )
    assert result.indices.tolist() == [[0]]
    assert result.status.item() == 8
