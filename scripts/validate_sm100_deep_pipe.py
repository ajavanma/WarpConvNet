# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Residual-case check for the sm_100 deep-pipe forward tiles (ids 1000-1009).

**This is not the primary evidence that these kernels are correct.** Their author
has already validated all ten on a B200 — 136/136 numeric edge-case rows over a
matrix covering C tails, k=1, k=2, stride 2, groups 2/4, tiny N, N=500k, two
geometries x two dtypes x two seeds, plus ``compute-sanitizer racecheck``
reporting 0 hazards **at C=64 and C=256** (the "occupancy pivot / tiles 1006-1009"
commit; upstream chrockey/WarpConvNet@96e77fe, squashed into the sm_100 feature
commit here). Read that as the bring-up
gate.

Quote the scope, not just the verdict: the sibling "pool order from the iteration-2
measurement" commit (chrockey@5b352f6) says only "racecheck: 0 hazards", and reading
that sentence instead of the scoped one is how two reviewers concluded for
two rounds that ``num_k_tiles == 1`` had been racechecked when it had not. The
unscoped sentence is not wrong, merely less specific — which is exactly why
nothing about reading it feels like missing something.

This script covers the two structural cases that matrix does not reach, and
re-confirms both after our tile-metadata refactor:

1. **The partial-prolog path** (Phase 1, the primary payload). It triggers when
   ``total_iters < NumStages - 1``, with ``total_iters = num_k_tiles * num_active``
   and ``num_k_tiles = ceil(C_in/tile_k)``. That is a property of how many *kernel
   offsets are active*, not of how many rows there are — "tiny N" shrinks rows and
   leaves ``num_active`` at 27. So the sweep drives ``num_active`` down directly
   with strided voxel lattices: a stride of 2 on an axis removes every neighbour
   offset on that axis, giving ``num_active`` of exactly 1 (2,2,2) or 3 (1,2,2) at
   any N. At the deepest pool-reachable depth (NS=6, ``num_k_tiles=2`` at C_in=64)
   the path needs ``num_active <= 2``, so only the sparse lattices reach it. The
   achieved ``num_active`` is printed per case and a tile whose partial-prolog path
   was never reached is reported as NOT EXERCISED and fails the run.
2. **The ``num_k_tiles == 1`` configuration** (Phase 3b), reached only at C_in=32.

Phase structure follows what autotune can actually select. The pools are
``_AB_MASK_SM100_WIDE = (1009, 1006, 1002, 1001)`` and
``_AB_MASK_SM100_NARROW = (1007, 1008, 1000)``, so:

* **pool-reachable = {1000, 1001, 1002, 1006, 1007, 1008, 1009}** — max reachable
  ``NumStages`` is 6, not 10.
* **pool-unreachable (pin-only) = {1003, 1004, 1005}** — deliberately excluded
  measured losers (1004 at NS=10 was a 1.43x regression versus 1000; 1005 was
  worst in every cell). Phase 3a covers them only behind ``--include-unpooled``.

Phase 1 is "unpinned" in the sense that it covers exactly the pool-reachable set
and reports which tile the real adaptive pool selects at this shape. It still
executes each of the seven by explicit tile_id, because that is the only way to
cover a specific tile — autotune picks one winner, not seven. Both facts are
reported; neither is presented as the other.

The four anti-vacuous-pass gates are unchanged and are the durable part of this
script. Each of these failure modes otherwise produces a run that prints PASS
having verified nothing:

* **Gate 1 (arch).** ``compile_archs=(100,)`` is an EXACT pin. ``_tile_arch_allowed``
  only extends a ``compile_archs`` entry to binary-compatible minor revisions when
  the tuple has 2+ entries, so sm_103 does NOT inherit sm_100 here and sm_120 is a
  different major family. On a GB300 or an RTX PRO 6000 all ten tiles are filtered
  out of ``_get_tiles`` and the sweep would report PASS over an empty set. Only
  sm_100 exactly is accepted.
* **Gate 2 (presence).** All ten ids must be in the FULL forward metadata index AND
  pass ``tile_launch_rejection``. A branch that did not build them, or a tier gate
  that dropped them during a warpgemm reimport, otherwise yields a
  smaller-but-still-green run. Deliberately NOT checked against ``_get_tiles``:
  that passes ``active_only=True`` and keeps only ``tier="production"``, and all
  ten of these are ``tier="experimental"`` on purpose — so on a real sm_100 GB200
  it reported all ten as arch-filtered-out when the actual reason was tier.
  "In the production pool" is not "usable".
* **Gate 3 (reference signal).** Every fp64 reference is proven finite and non-zero
  *before* any tile verdict. This is not hypothetical here:
  ``_reference_has_signal`` in
  ``warpconvnet/nn/functional/sparse_conv/detail/autotune.py:443`` is a *designed*
  fail-open — "The reference algo (explicit_gemm) overflows identically to every
  candidate ... Treat it as no-signal and fail open." Correct for the autotuner,
  fatal for a gate. Note ``validate_tiles_on_device.py`` does
  ``if ref_max <= 0: continue``, a silent skip; this script does not inherit it.
* **Gate 4 (execution coverage).** A binding refusal on a shape the tile declares
  ADMISSIBLE is a FAIL, not a coverage note. Admissibility is read per tile from
  ``handles_c_in``/``handles_c_out``, so a shape a tile legitimately rejects is
  reported as uncovered — never as a defect, and never as a pass.

Guiding rule throughout: assert what DID run, never what failed to be rejected.

**Why the phases use different N.** Numeric defaults to ``--n 200000`` because
corruption has to reach the output tensor. Racecheck does not — it reports from
instrumented shared-memory accesses, so a hazard is flagged whether or not it
corrupted anything, and ``sanitize_tile.py`` documents the ~100x slowdown that
makes 200k impractical. Racecheck defaults to ``--racecheck-n 20000``. Since the
lattices fix ``num_active`` independently of N, the smaller N costs no
partial-prolog coverage. The difference is stated, not silently applied.

Exit codes: 0 all gates and phases passed; 1 numeric/coverage/racecheck failure;
2 wrong arch; 3 tiles missing/rejected; 4 dead reference; 5 tooling/argument;
6 script and imported ``warpconvnet`` in different repos (Gate 0).

    python scripts/validate_sm100_deep_pipe.py                    # phases 1, 2, 3b
    python scripts/validate_sm100_deep_pipe.py --include-unpooled # + 1003/1004/1005
    python scripts/validate_sm100_deep_pipe.py --repeats 10
    python scripts/validate_sm100_deep_pipe.py --self-test-imports  # wiring only
"""

import argparse
import importlib
import itertools
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

import warpconvnet  # noqa: F401
from warpconvnet.nn.functional.sparse_conv.detail import tile_metadata as tm

TILES = tuple(range(1000, 1010))
# Expected pool membership, from algo_params_sm100.py. Only a cross-check: the
# live pool is read from candidate_pool() at the shape under test, because a
# hard-coded set that silently disagrees with the real pool is how a phase ends
# up "covering" tiles autotune can never pick.
POOL_EXPECTED = (1000, 1001, 1002, 1006, 1007, 1008, 1009)
PIN_ONLY_EXPECTED = (1003, 1004, 1005)
K1_PROBE_TILE = 1004  # deepest pipeline; the num_k_tiles==1 question (Phase 3b)
REQUIRED_ARCH = 100  # exact; see Gate 1
OPS = ("forward", "dgrad", "wgrad")

EXIT_OK, EXIT_WRONG, EXIT_ARCH, EXIT_TILES, EXIT_REF, EXIT_TOOLING = 0, 1, 2, 3, 4, 5
# Script and imported package in different repos -- see gate_package_provenance().
EXIT_PROVENANCE = 6
# --self-test-imports deliberately exits NONZERO. It launches zero kernels, so a
# CI job that reads only the exit code must never be able to score it as a pass.
# A printed banner is a comment; the exit code is the boundary. This whole script
# exists because five separate gates in this feature were found reporting success
# for work they never did — do not "fix" this to 0.
EXIT_SELFTEST = 64

# Lattice strides and the num_active each yields for a 3x3x3 kernel. Stride 2 on
# an axis means no voxel ever has a +-1 neighbour there, so all nine offsets with
# a nonzero component on that axis stay empty:
# num_active = prod(1 if stride >= 2 else 3). Measured on device at 20k points:
# (2,2,2) -> 1, (1,2,2) -> 3, (1,1,2) -> 9, dense random -> 27. Independent of N.
ACTIVITIES = {"sparse-1": (2, 2, 2), "sparse-3": (1, 2, 2), "dense": None}
DEFAULT_ACTIVITIES = "sparse-1,sparse-3,dense"

# racecheck's wording for skipped accesses has changed across toolkit releases,
# so match a family of phrasings. A skipped access is one racecheck did not
# check, and it is indistinguishable from clean in the summary line — the exact
# vacuous pass this script exists to prevent. Same for the warning count:
# "Maximum number of hazards reached" truncates the report, so warnings>0 makes
# the error count a floor rather than an answer.
_SKIPPED_N = re.compile(r"(\d+)\s+(?:\w+\s+){0,2}(?:were\s+|was\s+)?skipp?ed", re.I)
_SKIP_LINE = re.compile(r"skipp?(?:ed|ing)|lost due to|not instrumented|could not be checked", re.I)


def _scripts_dir(explicit=None):
    """Locate the repo's ``scripts/`` directory.

    On the collaborator's branch this file lives there, so its own directory is
    the answer. The other candidates let it be exercised from a staging location
    before it is committed, resolving via the installed (editable) package rather
    than guessing.
    """
    cands = []
    if explicit:
        cands.append(Path(explicit))
    here = Path(__file__).resolve().parent
    cands += [here, here.parent / "scripts"]
    if os.environ.get("WCN_SCRIPTS_DIR"):
        cands.append(Path(os.environ["WCN_SCRIPTS_DIR"]))
    cands.append(Path(warpconvnet.__file__).resolve().parent.parent / "scripts")
    cands += [p / "scripts" for p in Path.cwd().resolve().parents]
    for c in cands:
        if (c / "validate_tiles_on_device.py").is_file() and (c / "racecheck_tiles.py").is_file():
            return c.resolve()
    raise SystemExit(
        "cannot locate the repo scripts/ directory (need validate_tiles_on_device.py "
        "and racecheck_tiles.py). Pass --scripts-dir."
    )


def _load(scripts_dir):
    """Import the sibling scripts as modules.

    They are scripts, not a package, so the path goes on ``sys.path`` and they are
    imported by name. Everything load-bearing is reused rather than re-derived:
    ``run_tile``/``RTOL`` and the geometry/reference helpers for the numeric
    phases, and ``racecheck_tiles``' repo cwd, hazard-buffer env var and
    summary regex — the last carrying a fix (singular "1 warning") that a fresh
    copy gets wrong in exactly the high-hazard cases that matter.
    """
    sys.path.insert(0, str(scripts_dir))
    return (
        importlib.import_module("validate_tiles_on_device"),
        importlib.import_module("racecheck_tiles"),
    )


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------


def _coords(n, stride):
    """Voxel coordinates: a strided lattice, or a dense random interior."""
    if stride is None:
        # Same recipe as validate_tiles_on_device.build, so the dense case here
        # is comparable with that harness's results.
        torch.manual_seed(0)
        side = max(8, int((n * 3) ** (1 / 3)) + 1)
        return torch.unique(
            torch.randint(0, side, (n * 2, 3), device="cuda", dtype=torch.int32), dim=0
        )[:n]
    side = int(math.ceil(n ** (1 / 3)))
    g = torch.arange(side, device="cuda", dtype=torch.int32)
    grid = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)[:n]
    return grid * torch.tensor(stride, device="cuda", dtype=torch.int32)


def _problem(vtod, n, c_in, c_out, stride, refs=True):
    """One problem plus its fp64 references, for a given lattice stride.

    Mirrors ``validate_tiles_on_device.build``, which cannot be called directly
    because it hard-codes dense random coordinates and controlling ``num_active``
    is the entire point here. Every helper and constant comes out of that
    module's namespace rather than being re-imported, so the reference
    construction cannot drift from the harness it is meant to match — including
    the ``4096/n_out`` grad scaling, which exists because wgrad accumulates one
    term per matched row and at N=200k with unit inputs the true result (~1e5)
    overflows fp16 in EVERY tile. Without it the sweep reports inf everywhere and
    hides the real failures.
    """
    c = _coords(n, stride)
    vox = vtod.Voxels([c], [torch.ones(c.shape[0], c_in, device=vtod.DEV, dtype=torch.float32)])
    _, _, kmap = vtod.generate_output_coords_and_kernel_map(
        input_sparse_tensor=vox,
        kernel_size=vtod.KERNEL_SIZE,
        kernel_dilation=(1, 1, 1),
        stride=(1, 1, 1),
        generative=False,
        transposed=False,
    )
    n_out = vox.feature_tensor.shape[0]
    x = torch.ones(n_out, c_in, device=vtod.DEV, dtype=torch.float32)
    w = torch.ones(vtod.KERNEL_VOLUME, c_in, c_out, device=vtod.DEV, dtype=torch.float32)
    # Row-varying grad_output: a constant grad cannot detect a scatter landing on
    # the wrong row.
    row = (torch.arange(n_out, device=vtod.DEV, dtype=torch.float32) % 8) / 8.0
    col = torch.arange(c_out, device=vtod.DEV, dtype=torch.float32) / max(c_out, 1)
    g = (row.unsqueeze(1) + col.unsqueeze(0)) / 2.0 * (4096.0 / max(n_out, 1))

    off = kmap.offsets
    prob = {
        "x": x,
        "w": w,
        "g": g,
        "kmap": kmap,
        "n_out": n_out,
        "num_active": int(((off[1:] - off[:-1]) > 0).sum().item()),
    }
    if not refs:
        # The sanitizer driver launches one kernel and checks nothing numerically;
        # an fp64 explicit-GEMM reference there would dominate the runtime of an
        # already ~100x-instrumented run for no benefit.
        return prob
    prob["forward"] = vtod._explicit_gemm_forward_logic(
        x.double(), w.double(), kmap, n_out, torch.float64
    )
    prob["dgrad"], prob["wgrad"] = vtod._explicit_gemm_backward_logic(
        g.double(), x.double(), w.double(), kmap, torch.float64, torch.device(vtod.DEV)
    )
    return prob


def prolog_budget(meta, c_in):
    """``(num_k_tiles, max num_active)`` that still reaches the partial prolog.

    Derived from metadata rather than hard-coded, so it stays right if
    ``num_stages``/``tile_k`` change under us. Both intermediates are printed, so
    a wrong assumption shows up as a visibly wrong budget instead of as silent
    miscoverage.
    """
    k_tiles = max(1, math.ceil(c_in / meta.tile_k))
    return k_tiles, max(0, math.ceil((meta.num_stages - 1) / k_tiles) - 1)


def hits_partial_prolog(meta, c_in, num_active):
    k_tiles, _ = prolog_budget(meta, c_in)
    return k_tiles * num_active < meta.num_stages - 1


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def gate_package_provenance():
    """The imported ``warpconvnet`` must live in the same repo as this script.

    ``python scripts/validate_sm100_deep_pipe.py`` puts ``scripts/`` on
    ``sys.path[0]`` -- NOT the repo root -- so ``import warpconvnet`` resolves via
    the installed (editable) package, which on a multi-worktree checkout points at
    a DIFFERENT tree than the one you are standing in. Measured on this
    workstation: run from the merge worktree it imported the main worktree's
    package and read cache version 17.0 instead of the branch's 18.0.

    That is not cosmetic here. ``_scripts_dir`` falls back to
    ``warpconvnet.__file__``'s repo to find the sibling helpers, so a mismatch
    propagates into which ``validate_tiles_on_device`` gets loaded, and Gate 2
    would then report tile presence for a tree the operator is not testing --
    passing or failing for reasons that have nothing to do with their branch.

    Fix at the call site, not here: run ``PYTHONPATH=$(git rev-parse --show-toplevel)``
    or ``pip install -e .`` in the checkout under test.
    """
    pkg_repo = Path(warpconvnet.__file__).resolve().parent.parent
    script_repo = Path(__file__).resolve().parent.parent
    if pkg_repo == script_repo:
        print(f"GATE 0 provenance: PASS — warpconvnet imported from {pkg_repo}")
        return None
    return (
        f"GATE 0 provenance: FAIL — this script and the warpconvnet it imported are in\n"
        f"  different repos, so the run would validate a tree you are not testing.\n"
        f"    script : {script_repo}\n"
        f"    package: {pkg_repo}\n"
        f"  sys.path[0] is scripts/, not the repo root, so `import warpconvnet` fell\n"
        f"  through to the installed editable package. Re-run as:\n"
        f"    PYTHONPATH=\"{script_repo}\" {Path(sys.executable).name} {Path(__file__).name}\n"
        f"  or pip install -e . inside {script_repo}."
    )


def gate_arch():
    arch = tm._get_device_arch()
    if arch == REQUIRED_ARCH:
        print(f"GATE 1 arch: PASS — device is sm_{arch}")
        return None
    where = "no CUDA device" if arch is None else f"sm_{arch}"
    return (
        f"GATE 1 arch: FAIL — this script requires sm_{REQUIRED_ARCH} exactly, found {where}.\n"
        f"  Tiles {TILES[0]}-{TILES[-1]} are compile_archs=(100,), an EXACT pin:\n"
        f"  _tile_arch_allowed() only extends compile_archs to binary-compatible minor\n"
        f"  revisions when the tuple has 2+ entries, so sm_103 does NOT inherit sm_100,\n"
        f"  and sm_120 is a different major family. On this device all ten tiles are\n"
        f"  filtered out of _get_tiles('forward'), so the run would have launched zero\n"
        f"  kernels and printed PASS. NOTHING WAS VERIFIED. Run this on a B200."
    )


def gate_tiles():
    """All ten ids must be present in metadata AND launch-authorized. Partial sets abort.

    Presence is checked against ``_metadata_index`` (the FULL per-op registry),
    not ``_get_tiles``. ``_get_tiles`` passes ``active_only=True``, which keeps
    only ``tier="production"`` — and all ten deep-pipe tiles are
    ``tier="experimental"`` by design, reached through the ``algo_params_sm100``
    override pools rather than the adaptive pool. Checking the active list
    therefore reported all ten as "NOT in arch-filtered tile list" on a real
    sm_100 GB200, where the true reason was tier, not arch. Verified: they are
    absent from ``_get_tiles('forward', filter_arch=False)`` and present in
    ``known_tile_ids('forward')``.

    Authorization is ``tile_launch_rejection``, which is the authority that
    actually gates a launch — it consults the full index, the arch pin and
    ``_LAUNCHABLE_BACKENDS``. It is also what
    ``test_sm100_deep_pipe_fwd_launchable_only_on_sm100`` asserts against.

    Same conflation this script exists to catch: "in the production pool" is not
    "usable", and for an experimental tile the two genuinely differ.
    """
    index = tm._metadata_index("forward")
    missing, rejected = [], []
    for tid in TILES:
        meta = index.get(tid)
        if meta is None:
            missing.append((tid, "absent from forward metadata entirely (not built on this branch?)"))
        elif (why := tm.tile_launch_rejection("forward", tid)) is not None:
            rejected.append((tid, f"{why} [compile_archs={meta.compile_archs}, backend={meta.backend!r}]"))
    if missing or rejected:
        lines = [
            f"GATE 2 tiles: FAIL — {len(missing) + len(rejected)} of {len(TILES)} deep-pipe "
            f"tiles are not usable; refusing to run a partial sweep."
        ]
        lines += [f"    tile {t}: NOT in forward metadata — {w}" for t, w in missing]
        lines += [f"    tile {t}: launch-rejected — {w}" for t, w in rejected]
        lines.append("  A smaller sweep would still print PASS. Fix the build/branch first.")
        return None, "\n".join(lines)
    print(f"GATE 2 tiles: PASS — all {len(TILES)} ids present and launchable")
    return index, None


def gate_references(problems):
    """Prove every reference carries signal before any tile verdict is formed.

    Deliberately NOT ``validate_tiles_on_device``'s ``if ref_max <= 0: continue``.
    A dead reference silently turns the whole run into an assertion-free no-op —
    the same fail-open shape as ``autotune._reference_has_signal``, which is
    correct there and fatal here.
    """
    dead = []
    for key, prob in problems.items():
        for op in OPS:
            ref = prob[op].float()
            if not torch.isfinite(ref).all().item():
                dead.append((key, op, "non-finite (inf/NaN)"))
            elif ref.abs().max().item() <= 0:
                dead.append((key, op, "identically zero"))
    if dead:
        lines = ["GATE 3 references: FAIL — no usable reference signal for some (case, op):"]
        lines += [f"    {a} C {ci}->{co}  {op}: {w}" for (a, ci, co), op, w in dead]
        lines.append(
            "  A comparison against such a reference cannot distinguish a correct tile "
            "from one that writes garbage."
        )
        return "\n".join(lines)
    print(
        f"GATE 3 references: PASS — {len(problems)} cases x {len(OPS)} ops, "
        f"all fp64 references finite and non-zero"
    )
    return None


# ---------------------------------------------------------------------------
# Pool reachability
# ---------------------------------------------------------------------------


def pool_reachable(c_in, c_out, n, kv=27):
    """Deep-pipe ids the adaptive pool can actually select at this shape.

    Read live from ``candidate_pool`` rather than trusting ``POOL_EXPECTED``: a
    phase that "covers" tiles autotune can never pick is exactly the kind of
    partial coverage this script is supposed to make visible. Returns
    ``(reachable, notes)``; a disagreement with the documented pool is a note,
    not a silent correction.
    """
    notes = []
    try:
        from warpconvnet.nn.functional.sparse_conv.detail.algo_params import candidate_pool

        pool = candidate_pool("AB", "auto", c_in, c_out, kv, num_in_coords=n)
        live = {
            p["tile_id"]
            for algo, p in pool
            if algo == "mask_gemm" and p.get("tile_id") in TILES
        }
    except Exception as e:  # pool API moved or shape unsupported
        notes.append(
            f"could not read the live candidate pool ({type(e).__name__}: {e}); "
            f"falling back to the documented pool {list(POOL_EXPECTED)} — pool "
            f"membership itself is therefore UNVERIFIED"
        )
        return sorted(POOL_EXPECTED), notes
    if live != set(POOL_EXPECTED):
        # POOL_EXPECTED is the UNION of the WIDE and NARROW bands. Only one band is
        # live at any shape: _ab_mask_pool bands on C_out (>=128 -> WIDE
        # (1009,1006,1002,1001), else NARROW (1007,1008,1000)) per the "band the
        # deep-pipe tiles on C_out" change (chrockey@9694710), which
        # fixed a banding bug by keying on C_out alone. So a subset here is normal
        # and does NOT mean anything is stale -- an earlier wording said it did,
        # which would have sent someone "fixing" a correct algo_params_sm100.
        wide = c_out >= 128
        band = "WIDE" if wide else "NARROW"
        other, hint = ("NARROW", f"--channels {c_in}:64") if wide else ("WIDE", f"--channels {c_in}:128")
        missing = sorted(set(POOL_EXPECTED) - live)
        notes.append(
            f"live pool at C={c_in}->{c_out} is {sorted(live)} ({band} band; pools are "
            f"banded on C_out by _ab_mask_pool). Union of both bands is "
            f"{list(POOL_EXPECTED)}, so {missing} are NOT covered at this shape — "
            f"run the {other} band ({hint}) to reach them, or pass both shapes in one "
            f"run. Using the live set."
        )
    return sorted(live), notes


def observe_pool_winner(vtod, prob, c_in, c_out, reachable):
    """Report which candidate the adaptive pool actually picks at this shape.

    Phase 1 executes each of the seven by explicit tile_id because autotune picks
    one winner, not seven. That is a different claim from "the unpinned path runs
    a deep-pipe tile", so the winner is measured and reported rather than assumed
    in either direction.
    """
    try:
        from warpconvnet.nn.functional.sparse_conv.detail.algo_params import candidate_pool
        from warpconvnet.nn.functional.sparse_conv.detail.autotune import (
            _run_forward_benchmarks,
        )

        pool = candidate_pool("AB", "auto", c_in, c_out, 27, num_in_coords=prob["n_out"])
        results = _run_forward_benchmarks(
            prob["x"].half(),
            prob["w"].half(),
            prob["kmap"],
            prob["n_out"],
            torch.float16,
            custom_params=pool,
        )
        if not results:
            return "adaptive pool produced no viable candidate at this shape"
        algo, params, _ = results[0]
        tid = params.get("tile_id")
        if algo == "mask_gemm" and tid in reachable:
            return f"adaptive pool selects mask_gemm tile {tid} (a deep-pipe tile)"
        return (
            f"adaptive pool selects {algo} params={params} — NOT a deep-pipe tile, so the "
            f"unpinned production path does not exercise {reachable} at this shape; "
            f"Phase 1's coverage below comes from explicit tile_id dispatch only"
        )
    except Exception as e:
        return f"could not observe the pool winner ({type(e).__name__}: {e}) — UNVERIFIED"


# ---------------------------------------------------------------------------
# Numeric sweep (shared by phases 1, 3a, 3b)
# ---------------------------------------------------------------------------


def numeric_sweep(vtod, index, tiles, problems, cases, per_op, repeats, label):
    """Sweep ``tiles`` over ``cases``. Returns ``(failed, notes, prolog_hits)``."""
    rtol = vtod.RTOL
    failed, notes = False, []
    prolog_hits = {t: None for t in tiles}
    for op in OPS:
        op_tiles = [t for t in tiles if t in per_op[op]]
        if not op_tiles:
            notes.append(
                f"{label} {op}: none of {tiles} are registered/launchable for {op} — "
                f"NOT verified for this op (expected for forward-only kernels)"
            )
            continue
        # A PARTIAL op is the quiet version of the same hole: 4 of 7 ids
        # registered for dgrad still prints "dgrad PASS". Name the absentees.
        if absent := [t for t in tiles if t not in op_tiles]:
            notes.append(f"{label} {op}: ids {absent} not registered for {op} — NOT verified")
        print(f"    {op}: {len(op_tiles)}/{len(tiles)} tiles x {len(cases)} cases x {repeats} reps")
        bad, refused = {}, {}
        for tid, key in itertools.product(op_tiles, cases):
            meta, prob = index[tid], problems[key]
            act, c_in, c_out = key
            if not (meta.handles_c_in(c_in) and meta.handles_c_out(c_out)):
                # The tile declares it cannot take this shape, so a refusal is
                # correct behaviour: uncovered, not defective.
                notes.append(
                    f"{label} {op} tile {tid}: C {c_in}->{c_out} inadmissible "
                    f"(min C_in={meta.min_per_group_c_in}, C_out={meta.min_per_group_c_out}) "
                    f"— NOT verified there"
                )
                continue
            iters = prolog_budget(meta, c_in)[0] * prob["num_active"]
            if op == "forward" and hits_partial_prolog(meta, c_in, prob["num_active"]):
                prev = prolog_hits[tid]
                prolog_hits[tid] = iters if prev is None else min(prev, iters)
            ref = prob[op]
            ref_max = ref.abs().max().item()  # Gate 3 proved this is > 0
            worst, runs_wrong, ran = 0.0, 0, False
            for _ in range(repeats):
                try:
                    out = vtod.run_tile(op, tid, prob)
                except RuntimeError as e:
                    refused.setdefault(tid, []).append((key, f"{type(e).__name__}: {e}"))
                    break
                if out is None:
                    refused.setdefault(tid, []).append((key, "binding returned no tensor"))
                    break
                ran = True
                rel = (out.float() - ref.float()).abs().max().item() / ref_max
                worst = max(worst, rel)
                runs_wrong += rel > rtol
            if ran and worst > rtol:
                bad.setdefault(tid, []).append((key, worst, runs_wrong, iters))
        if refused:
            # Gate 4: on an ADMISSIBLE shape a refusal is a defect, not a note.
            failed = True
            print(
                f"      FAIL (coverage) — {len(refused)} tiles refused an ADMISSIBLE shape. "
                f"The tile declares it handles the shape, so an unrun tile is an "
                f"unverified tile, not a pass."
            )
            for tid, rows in sorted(refused.items()):
                for (act, ci, co), why in rows:
                    print(f"        tile {tid} {act:<9} C {ci}->{co}: did not execute: {why}")
        if bad:
            failed = True
            print(f"      FAIL (numeric) — {len(bad)} tiles produce wrong results:")
            for tid, rows in sorted(bad.items()):
                # Wrong on every repeat is a logic bug; wrong on some is a race.
                kind = "DETERMINISTIC" if all(r == repeats for *_, r, _ in rows) else "INTERMITTENT"
                print(f"        tile {tid:>4}  {index[tid].kernel_struct}   [{kind}]")
                for (act, ci, co), rel, runs_wrong, iters in rows:
                    path = (
                        "PARTIAL-PROLOG"
                        if iters < index[tid].num_stages - 1
                        else "full prolog"
                    )
                    print(
                        f"            {act:<9} C {ci:>4}->{co:<4} total_iters={iters:<4} "
                        f"[{path}]  max_rel={rel:<10.4g} wrong in {runs_wrong}/{repeats}"
                    )
        if not bad and not refused:
            print(f"      PASS — every admissible (tile, case) correct on all {repeats} repeats")
    return failed, notes, prolog_hits


def report_prolog_coverage(index, problems, cases, tiles, prolog_hits):
    """Prove the partial-prolog path was reached per tile. Unreached is a FAIL.

    The sparse lattices exist solely to hit this path. If they did not, the run
    exercised the well-fed steady state only — which the author's B200 matrix
    already covers — and a green result here would add nothing while looking like
    it added everything.
    """
    print("\n  partial-prolog coverage (total_iters < NumStages-1):")
    missed = []
    for tid in tiles:
        meta = index[tid]
        if (hit := prolog_hits.get(tid)) is not None:
            print(
                f"    tile {tid:>4} NS={meta.num_stages:<3} EXERCISED "
                f"(best total_iters={hit} < {meta.num_stages - 1})"
            )
            continue
        adm = [k for k in cases if meta.handles_c_in(k[1]) and meta.handles_c_out(k[2])]
        k_tiles, budget = prolog_budget(meta, adm[0][1]) if adm else (0, 0)
        got = min((problems[k]["num_active"] for k in adm), default=None)
        if not adm:
            # The tile rejects every shape tested -- e.g. min_per_group_c_out=128
            # against a C_out=64 sweep. Gate 4's rule applies: a shape a tile
            # legitimately rejects is UNCOVERED, never a defect. Reporting it as
            # "needed <=0 at num_k_tiles=0" (the empty-adm fallback) named a budget
            # that was never computed and read as a coverage failure.
            print(
                f"    tile {tid:>4} NS={meta.num_stages:<3} NOT ADMISSIBLE at any tested "
                f"shape — needs C_in>={meta.min_per_group_c_in}, C_out>="
                f"{meta.min_per_group_c_out}; tested {[f'{c[1]}->{c[2]}' for c in cases]}. "
                f"Uncovered here, NOT a failure. Add an admissible shape to --channels."
            )
            continue
        if budget == 0:
            # Not a coverage gap — an arithmetic impossibility. The path needs
            # total_iters = num_k_tiles * num_active < NumStages-1, so a budget of
            # 0 demands num_active <= 0, i.e. no active offsets, i.e. no work at
            # all. No input can produce it. Measured case: tile 1009 (NS=3) at
            # C_in=64, num_k_tiles=2 -> needs num_active<=0.
            #
            # Reported loudly rather than passed silently, and deliberately NOT
            # folded into the FAIL: failing here would demand an input that cannot
            # exist, and a gate nobody can satisfy gets disabled rather than fixed.
            # The distinction that keeps this honest: budget==0 is impossible,
            # budget>=1 unreached is an inadequate input construction and still FAILs.
            print(
                f"    tile {tid:>4} NS={meta.num_stages:<3} STRUCTURALLY UNREACHABLE — "
                f"needs num_active<=0 at num_k_tiles={k_tiles} (C_in={adm[0][1]}), so no "
                f"input reaches it. For NS={meta.num_stages} the partial prolog requires "
                f"num_k_tiles==1, i.e. C_in<={meta.tile_k}, which this tile declares "
                f"inadmissible (min_per_group_c_in={meta.min_per_group_c_in}). The path is "
                f"dead for this tile in every admissible configuration."
            )
            continue
        print(
            f"    tile {tid:>4} NS={meta.num_stages:<3} *** NOT EXERCISED *** "
            f"(best num_active={got}, needed <={budget} at num_k_tiles={k_tiles})"
        )
        missed.append(tid)
    if missed:
        print(
            f"    FAIL — partial-prolog path never reached for {missed}. That path is this "
            f"script's primary payload; without it the run only re-covers ground the "
            f"author's B200 matrix already covers. Add a sparser activity config "
            f"(--activities) or check the num_stages/tile_k the budget came from."
        )
    return bool(missed)


# ---------------------------------------------------------------------------
# Racecheck
# ---------------------------------------------------------------------------


_ACTIVE_RE = re.compile(r"num_active=(\d+)")


def _driver_launch(args):
    """Launch exactly one tile, for running under compute-sanitizer.

    This script re-invokes itself here rather than shelling out to
    ``sanitize_tile.py``, for one reason: that driver builds dense random
    coordinates and has no way to express a strided lattice, so using it would
    racecheck a ``num_active=27`` problem while the report claimed sparse
    partial-prolog coverage — precisely the vacuous pass this script exists to
    prevent. Re-invoking ourselves means the geometry under the sanitizer comes
    from the same ``_coords`` call the numeric phase used, so the two phases
    provably exercise the same thing.

    ``sanitize_tile.py``'s contract is kept exactly — one launch, no fp64
    reference work, an ``OK `` sentinel line on success — so ``racecheck_tiles``'
    output parsing applies unchanged. Its rationale applies unchanged too:
    racecheck instruments every shared-memory access at roughly 100x, and a
    hazard is reported from the instrumented accesses whether or not it corrupted
    this particular run.
    """
    vtod, _ = _load(_scripts_dir(args.scripts_dir))
    c_in, c_out = (int(v) for v in args.channels.split(":"))
    prob = _problem(vtod, args.n, c_in, c_out, ACTIVITIES[args.driver_activity], refs=False)
    torch.cuda.synchronize()
    vtod.run_tile(args.driver_op, args.tile, prob)
    torch.cuda.synchronize()
    # num_active goes in the sentinel so the sanitizer log itself records which
    # geometry ran — the parent asserts on it rather than trusting the flag it
    # passed down.
    print(
        f"OK {args.driver_op} tile={args.tile} C={c_in}->{c_out} "
        f"N_out={prob['n_out']} num_active={prob['num_active']}"
    )
    return EXIT_OK


def racecheck_one(rt, op, tile, cin, cout, n, activity, timeout):
    """One racecheck run. Returns ``(errors, warnings, skipinfo, active, hard)``.

    Not a call to ``racecheck_tiles.racecheck_one``: that returns only the error
    count, and this gate also requires warnings==0 and skipped==0, which need the
    raw output. Its constants ARE reused (hazard-buffer env var, summary regex,
    repo cwd) so the fiddly parts keep one source of truth.
    """
    cmd = [
        shutil.which("compute-sanitizer"),
        "--tool", "racecheck",
        "--racecheck-detect-level", "info",
        "--print-limit", "1",
        sys.executable, str(Path(__file__).resolve()),
        "--driver-launch",
        "--driver-op", op,
        "--tile", str(tile),
        "--channels", f"{cin}:{cout}",
        "--n", str(n),
        "--driver-activity", activity,
    ]  # fmt: skip
    env = dict(os.environ, NV_COMPUTE_SANITIZER_MAX_RACECHECK_HAZARDS=rt._MAX_HAZARDS)
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=rt.REPO, env=env
        )
    except subprocess.TimeoutExpired:
        return None, None, None, None, f"TIMEOUT after {timeout}s"
    out = r.stdout + r.stderr
    if "\nOK " not in "\n" + out:
        # The driver prints OK only after a successful synchronize; without it the
        # kernel never ran and a clean summary means nothing.
        tail = "\n".join(out.strip().splitlines()[-6:])
        return None, None, None, None, f"tile did not launch under sanitizer\n{tail}"
    m = rt._SUMMARY.search(out)
    if not m:
        return None, None, None, None, "no RACECHECK SUMMARY parsed — unknown, NOT clean"
    errors, warnings = int(m.group(1)), int(m.group(2))
    skipped, skip_lines = 0, []
    for line in out.splitlines():
        if not _SKIP_LINE.search(line):
            continue
        skip_lines.append(line.strip())
        hit = _SKIPPED_N.search(line)
        skipped = max(skipped, int(hit.group(1)) if hit else 1)
    active = _ACTIVE_RE.search(out)
    return errors, warnings, (skipped, skip_lines), (int(active.group(1)) if active else None), None


def racecheck_verdict(res):
    """``(ok, text)`` for one racecheck result under the 0/0/0 requirement."""
    errors, warnings, skipinfo, active, hard = res
    if hard is not None:
        return False, hard
    if active is None:
        # The launch succeeded but the sentinel did not carry the geometry, so
        # there is no evidence which problem was actually instrumented.
        return False, "driver did not report num_active — cannot confirm which geometry ran"
    skipped, skip_lines = skipinfo
    problems = []
    if errors:
        problems.append(f"{errors} hazards")
    if warnings:
        # Truncated report: the hazard count is a floor, not an answer.
        problems.append(f"{warnings} sanitizer warnings (report may be truncated)")
    if skipped:
        problems.append(f"{skipped} skipped (racecheck did not check those accesses)")
    if problems:
        detail = "".join(f"\n          {ln}" for ln in skip_lines[:3])
        return False, f"num_active={active}: {', '.join(problems)}{detail}"
    return True, f"num_active={active}: clean (0 hazards, 0 warnings, 0 skipped)"


def racecheck_sweep(rt, index, tiles, per_op, c_in, c_out, n, activities, timeout):
    failed, notes = False, []
    for op in ("forward", "dgrad"):
        op_tiles = [
            t
            for t in tiles
            if t in per_op[op] and index[t].handles_c_in(c_in) and index[t].handles_c_out(c_out)
        ]
        if not op_tiles:
            notes.append(f"racecheck {op}: no ids registered/admissible at this shape — NOT run")
            continue
        print(f"    {op}: {len(op_tiles)} tiles x {len(activities)} activities")
        for tid, act in itertools.product(op_tiles, activities):
            ok, text = racecheck_verdict(
                racecheck_one(rt, op, tid, c_in, c_out, n, act, timeout)
            )
            tag = f"tile {tid:>4} NS={index[tid].num_stages:<3} {act:<9}"
            print(f"      {tag}: {'' if ok else 'FAIL — '}{text}")
            failed |= not ok
    return failed, notes


# ---------------------------------------------------------------------------
# Phase 3b: tile 1004 at C_in=32 — the num_k_tiles == 1 question
# ---------------------------------------------------------------------------


def phase_3b(vtod, rt, index, n, c_out, racecheck_n, timeout):
    """Pin the deepest tile at C_in=32 and answer one question: is k_tiles==1 safe?

    C_in=32 gives ``num_k_tiles == 1``, which at NS=10 means nine distinct offsets
    in flight — the deepest cross-offset configuration these kernels can be put
    into, and the construct warpgemm 2819c2f DELETED from the generated mainloops
    ("pipelined/fused single-k-tile path: DELETE the cross-offset prefetch"). Its
    mainloop comment forbids reintroducing it; a hand-written kernel inherits no
    such guard.

    Nothing stops it reaching production except ``min_per_group_c_in=64``, which is
    NOT enforced at the binding (that checks ``mask_words``, not ``C_in``). So an
    explicit ``params={"tile_id": N}`` — what ``_execute_forward`` does, what a
    stale autotune cache entry does, and what ``validate_tiles_on_device`` itself
    does — walks straight into it. Note 64 is inherited convention, not a deep-pipe
    claim: 98 of 136 forward tiles carry it, including 2-stage tile 0, because it is
    the vectorized tile_k=32 loader shape (kVec 8 x 8). Do not read it as evidence
    that the author considered k_tiles==1 unsafe.

    RACECHECK IS THIS PHASE'S HEADLINE, NOT THE NUMERICS. The author's B200 run
    (the occupancy-pivot commit, chrockey@96e77fe) states its racecheck scope
    explicitly: "0 hazards at C=64 and C=256"
    — num_k_tiles 2 and 8. So k_tiles==1 was never racechecked. Its numerics may
    well be covered, because that harness pins past admissibility and its default
    channels include 32:32 and 32:64 — but numeric evidence is close to
    non-probative for THIS failure mode on THIS arch: 2819c2f records the same
    construct as "numerically latent on sm_100-lineage cubins for most tiles",
    surfacing only under racecheck. A clean numeric row here is what a present-and-
    silent hazard looks like. Compounding it, the quoted metrics (rdiff, cosine
    >= 0.9999985) are whole-tensor aggregates, and this repo has been fooled by
    exactly that at exactly this shape — tile 41 was wrong by up to 11x on ~500 of
    200,000 rows (0.25%), invisible to a mean-relative or cosine check.

    Runs LAST because a device-side assert here is sticky and would poison every
    later launch (see ``autotune._probe_context``), and dense because cross-offset
    prefetch needs multiple active offsets to prefetch across.
    """
    tid, c_in = K1_PROBE_TILE, 32
    meta = index[tid]
    k_tiles, _ = prolog_budget(meta, c_in)
    print(f"\n--- PHASE 3b: tile {tid} at C_in={c_in} — is num_k_tiles==1 safe? ---")
    print(
        f"  tile {tid} NS={meta.num_stages} tile_k={meta.tile_k} -> num_k_tiles={k_tiles} at "
        f"C_in={c_in}\n"
        f"  ({meta.num_stages - 1} offsets in flight; handles_c_in({c_in})="
        f"{meta.handles_c_in(c_in)}, so only a tile_id pin reaches this)\n"
        f"  Deciding: is TORCH_CHECK(C_in >= {meta.min_per_group_c_in}) belt-and-braces, "
        f"or load-bearing?"
    )
    if k_tiles != 1:
        return [], (
            f"INCONCLUSIVE — C_in={c_in} gives num_k_tiles={k_tiles}, not 1, for tile {tid} "
            f"(tile_k={meta.tile_k}). The configuration this probe targets was not reached."
        )
    try:
        prob = _problem(vtod, n, c_in, c_out, None)
    except RuntimeError as e:
        return [], f"INCONCLUSIVE — probe problem could not be built: {e}"
    ref = prob["forward"]
    if not torch.isfinite(ref).all().item() or ref.abs().max().item() <= 0:
        return [], "INCONCLUSIVE — forward reference carries no signal (Gate 3 rule)"

    findings = []
    try:
        out = vtod.run_tile("forward", tid, prob)
    except RuntimeError as e:
        print(f"  numeric: binding REFUSED — {type(e).__name__}: {e}")
        return [], (
            f"ALREADY GUARDED — the binding refuses tile {tid} at C_in={c_in}, so "
            f"TORCH_CHECK(C_in >= {meta.min_per_group_c_in}) would restate an existing "
            f"rejection (clearer message, no behaviour change)."
        )
    if out is None:
        print("  numeric: binding returned no tensor (refused)")
        return [], (
            f"ALREADY GUARDED — the binding refuses tile {tid} at C_in={c_in} (no tensor)."
        )
    rel = (out.float() - ref.float()).abs().max().item() / ref.abs().max().item()
    finite = torch.isfinite(out).all().item()
    wrong = (not finite) or rel > vtod.RTOL
    print(f"  numeric: EXECUTED, max_rel={rel:.4g}, finite={finite} -> "
          f"{'WRONG' if wrong else 'correct'}")
    if wrong:
        findings.append(f"wrong results at C_in={c_in} (max_rel={rel:.4g}, finite={finite})")
    ok, text = racecheck_verdict(
        racecheck_one(rt, "forward", tid, c_in, c_out, racecheck_n, "dense", timeout)
    )
    print(f"  racecheck: {'clean' if ok else 'DIRTY'} — {text}")
    if not ok:
        findings.append(f"racecheck not clean at C_in={c_in}: {text}")

    if findings:
        verdict = (
            f"LOAD-BEARING — tile {tid} executed at C_in={c_in} (num_k_tiles=1) and "
            f"misbehaved: {'; '.join(findings)}. TORCH_CHECK(C_in >= "
            f"{meta.min_per_group_c_in}) in the binding is required, not cosmetic — a "
            f"tile_id pin or a stale autotune cache entry can take this path today."
        )
    else:
        verdict = (
            f"BELT-AND-BRACES on this evidence — tile {tid} ran at C_in={c_in} "
            f"(num_k_tiles=1, {meta.num_stages - 1} offsets in flight) with correct numerics "
            f"and a clean racecheck. Weak evidence, deliberately stated as such: one clean "
            f"run of a configuration nothing declares support for is not a guarantee, and "
            f"min_per_group_c_in={meta.min_per_group_c_in} still has no stated rationale. "
            f"Adding the TORCH_CHECK remains the right call; this run just does not show it "
            f"is load-bearing."
        )
    print(f"\n  PHASE 3b CONCLUSION:\n    {verdict}")
    return findings, verdict


# ---------------------------------------------------------------------------


def _fail(msg, code):
    """Report a gate failure on stderr, ordered after whatever stdout printed.

    Without the flush the streams interleave and a banner can land below the
    failure that caused it, reading as if a later stage failed.
    """
    sys.stdout.flush()
    print(msg, file=sys.stderr)
    sys.stderr.flush()
    return code


def _activities(spec):
    acts = [a.strip() for a in spec.split(",") if a.strip()]
    if unknown := [a for a in acts if a not in ACTIVITIES]:
        raise SystemExit(f"unknown activity config(s) {unknown}; choose from {list(ACTIVITIES)}")
    return acts


def self_test_imports(args):
    """Wiring check only. Emphatically not a validation run — see the banner."""
    print("=" * 78)
    print("SELF-TEST (IMPORTS + WIRING ONLY) — THIS IS NOT A VALIDATION RUN.")
    print("It launches ZERO kernels and proves NOTHING about tiles 1000-1009.")
    print("Never use --self-test-imports as a CI gate for these kernels.")
    print("=" * 78)
    sd = _scripts_dir(args.scripts_dir)
    print(f"  scripts dir             : {sd}")
    vtod, rt = _load(sd)
    print(f"  validate_tiles_on_device: {vtod.__file__}  RTOL={vtod.RTOL}")
    print(f"  racecheck_tiles         : {rt.__file__}")
    # The single-launch driver is this file re-invoked with --driver-launch, not
    # sanitize_tile.py; see _driver_launch for why.
    print(f"  sanitizer driver        : {Path(__file__).resolve()} --driver-launch")
    for name in ("run_tile", "Voxels", "generate_output_coords_and_kernel_map", "KERNEL_SIZE"):
        assert hasattr(vtod, name), name
    for name in ("_SUMMARY", "_KERNEL", "_MAX_HAZARDS", "REPO"):
        assert hasattr(rt, name), name
    print(f"  device arch             : {tm._get_device_arch()} (required: {REQUIRED_ARCH})")
    print(f"  compute-sanitizer       : {shutil.which('compute-sanitizer')}")
    index = tm._metadata_index("forward")
    visible = {t.tile_id for t in tm._get_tiles("forward", filter_arch=True)}
    print(f"  deep-pipe ids visible   : {sorted(set(TILES) & visible)} of {list(TILES)}")
    print(f"  documented pool         : {list(POOL_EXPECTED)}  pin-only {list(PIN_ONLY_EXPECTED)}")
    if known := [t for t in TILES if index.get(t) is not None]:
        print("  metadata-derived plan (ids present in THIS tree only):")
        for t in known:
            m = index[t]
            k, b = prolog_budget(m, m.min_per_group_c_in)
            print(
                f"    tile {t}: NS={m.num_stages} tile_k={m.tile_k} "
                f"min_C_in={m.min_per_group_c_in} min_C_out={m.min_per_group_c_out} -> at "
                f"C_in={m.min_per_group_c_in} num_k_tiles={k}, partial prolog needs "
                f"num_active<={b}"
            )
    print(f"  activities              : {_activities(args.activities)}")
    print(f"  numeric N={args.n}  racecheck N={args.racecheck_n}  repeats={args.repeats}")
    print("SELF-TEST OK — imports and argument wiring resolve. STILL NOT A VALIDATION.")
    print(f"Exiting {EXIT_SELFTEST} (nonzero BY DESIGN) so no exit-code-only check")
    print("can score a zero-kernel wiring check as a validation pass.")
    return EXIT_SELFTEST


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repeats", type=int, default=3)
    # 200k is the numeric figure; racecheck does not inherit it (see docstring).
    p.add_argument("--n", type=int, default=200_000, help="numeric-phase rows")
    p.add_argument("--racecheck-n", type=int, default=20_000, help="racecheck-phase rows")
    p.add_argument("--channels", default="auto", help="'auto' (min admissible) or C_in:C_out,...")
    p.add_argument("--activities", default=DEFAULT_ACTIVITIES, help=f"of {list(ACTIVITIES)}")
    p.add_argument("--racecheck-activities", default=DEFAULT_ACTIVITIES)
    p.add_argument("--timeout", type=int, default=2400, help="per racecheck run, seconds")
    p.add_argument(
        "--include-unpooled",
        action="store_true",
        help=f"also sweep the pin-only tiles {list(PIN_ONLY_EXPECTED)} (Phase 3a). Off by "
        f"default: they are deliberately pool-excluded measured losers.",
    )
    p.add_argument("--skip-3b", action="store_true", help="skip the C_in=32 k_tiles==1 probe")
    p.add_argument("--scripts-dir", default=None, help="repo scripts/ dir (default: autodetect)")
    # Single-launch driver mode: how this script re-invokes itself under
    # compute-sanitizer. Not for interactive use.
    p.add_argument("--driver-launch", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--driver-op", default="forward", choices=OPS, help=argparse.SUPPRESS)
    p.add_argument("--driver-activity", default="dense", help=argparse.SUPPRESS)
    p.add_argument("--tile", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument(
        "--self-test-imports",
        action="store_true",
        help="check imports/wiring only; runs no kernels and validates nothing",
    )
    args = p.parse_args()

    if args.self_test_imports:
        return self_test_imports(args)
    if args.driver_launch:
        return _driver_launch(args)

    print(f"=== sm_{REQUIRED_ARCH} deep-pipe residual-case check: tiles "
          f"{TILES[0]}-{TILES[-1]} ===")
    print("  (the author has already B200-validated all ten; this covers the residual\n"
          "   structural cases — partial prolog, and num_k_tiles==1 — plus the gates)\n")

    if err := gate_package_provenance():
        return _fail(err, EXIT_PROVENANCE)
    if err := gate_arch():
        return _fail(err, EXIT_ARCH)
    index, err = gate_tiles()
    if err:
        return _fail(err, EXIT_TILES)

    min_c_in = max(index[t].min_per_group_c_in for t in TILES)
    min_c_out = max(index[t].min_per_group_c_out for t in TILES)
    if args.channels == "auto":
        # One shape per C_out BAND, not one shape overall. _ab_mask_pool bands on
        # C_out (>=128 -> WIDE, else NARROW), so any single shape reaches only one
        # band: the old auto value of (min_c_in, max min_c_out) = 64->128 covered
        # the WIDE band and hence 4 of the 10 tiles. It said so under NOT VERIFIED,
        # but a 4-of-10 run should not be what you get by default -- a reader who
        # does not reach that section reads a complete pass.
        #
        # Derived from the tiles' own declared minimums rather than hardcoded, so a
        # third band or a changed threshold is picked up automatically. Tiles that
        # reject the narrower shape are reported NOT ADMISSIBLE there (not failed)
        # and covered by the wider one; every tile accepts C_out=128 today.
        bands = sorted({index[t].min_per_group_c_out for t in TILES})
        channels = [(min_c_in, c) for c in bands]
        print(
            f"  --channels auto -> {[f'{a}->{b}' for a, b in channels]} "
            f"(one shape per C_out band; a single shape covers only one band)"
        )
    else:
        channels = [tuple(int(v) for v in pair.split(":")) for pair in args.channels.split(",")]
    if any(ci < min_c_in for ci, _ in channels):
        return _fail(
            f"refusing to run: --channels contains C_in below {min_c_in}, which every "
            f"deep-pipe tile declares inadmissible (handles_c_in is False). A pass or a "
            f"fail there says nothing about phases 1-2. C_in=32 is Phase 3b's job and is "
            f"labelled as such.",
            EXIT_TOOLING,
        )
    activities, race_acts = _activities(args.activities), _activities(args.racecheck_activities)

    # Fail before doing work if the required racecheck phase cannot run at all; a
    # missing sanitizer must never degrade into "numeric-only, reported as PASS".
    if not shutil.which("compute-sanitizer"):
        return _fail(
            "compute-sanitizer is not on PATH. Racecheck is REQUIRED for these deep-pipeline "
            "cp.async kernels (this repo deleted a cross-offset pipeline for a barrier-epoch "
            "hazard in 7249675 that numeric repeats could not see), so this is a failure, "
            "not a skip.",
            EXIT_TOOLING,
        )

    vtod, rt = _load(_scripts_dir(args.scripts_dir))
    c_in, c_out = channels[0]
    reachable, pool_notes = pool_reachable(c_in, c_out, args.n)
    unpooled = [t for t in TILES if t not in reachable]
    print(f"\n  pool-reachable at C={c_in}->{c_out}: {reachable}")
    print(f"  pin-only (not selectable by autotune): {unpooled}")
    for note in pool_notes:
        print(f"  NOTE: {note}")
    if not reachable:
        # Phases 1 and 2 both iterate the reachable set; an empty one would sweep
        # nothing and print PASS.
        return _fail(
            f"no deep-pipe tile is pool-reachable at C={c_in}->{c_out}, so phases 1 and 2 "
            f"would iterate an empty set and report success having launched nothing. "
            f"Expected {list(POOL_EXPECTED)}. Either the pool no longer contains them at "
            f"this shape, or algo_params_sm100.py is not on this branch — resolve that "
            f"before reading any result from this script.",
            EXIT_TILES,
        )
    for tid in sorted(TILES, key=lambda t: (-index[t].num_stages, t)):
        m = index[tid]
        k, b = prolog_budget(m, c_in)
        print(
            f"    tile {tid} NS={m.num_stages:<3} tile_k={m.tile_k:<4} "
            f"{'pool' if tid in reachable else 'PIN-ONLY':<8} at C_in={c_in}: "
            f"num_k_tiles={k}, partial prolog needs num_active<={b}"
        )

    cases = [(a, ci, co) for a in activities for ci, co in channels]
    problems = {k: _problem(vtod, args.n, k[1], k[2], ACTIVITIES[k[0]]) for k in cases}
    print("\n  activity configs (num_active drives the partial-prolog path):")
    for key in cases:
        prob = problems[key]
        print(
            f"    {key[0]:<9} C {key[1]:>4}->{key[2]:<4} N_out={prob['n_out']:>7} "
            f"num_active={prob['num_active']:>3}"
        )
    if err := gate_references(problems):
        return _fail(err, EXIT_REF)

    per_op = {op: [t for t in TILES if tm.tile_launch_rejection(op, t) is None] for op in OPS}
    ordered = lambda ts: sorted(ts, key=lambda t: (-index[t].num_stages, t))  # noqa: E731

    print(f"\n--- PHASE 1: partial-prolog targeting, pool-reachable tiles (N={args.n}) ---")
    print(f"  {observe_pool_winner(vtod, problems[cases[0]], c_in, c_out, reachable)}")
    p1_failed, p1_notes, p1_hits = numeric_sweep(
        vtod, index, ordered(reachable), problems, cases, per_op, args.repeats, "phase1"
    )
    p1_missed = report_prolog_coverage(index, problems, cases, ordered(reachable), p1_hits)

    p3a_failed, p3a_notes, p3a_missed = False, [], False
    if args.include_unpooled and unpooled:
        print(f"\n--- PHASE 3a: pin-only tiles {ordered(unpooled)} (opt-in) ---")
        p3a_failed, p3a_notes, p3a_hits = numeric_sweep(
            vtod, index, ordered(unpooled), problems, cases, per_op, args.repeats, "phase3a"
        )
        p3a_missed = report_prolog_coverage(index, problems, cases, ordered(unpooled), p3a_hits)

    # Free the fp64 references before racecheck; the subprocesses want the memory.
    problems.clear()
    torch.cuda.empty_cache()

    print(f"\n--- PHASE 2: racecheck, pool-reachable tiles (C={c_in}->{c_out}, "
          f"N={args.racecheck_n}) ---")
    print("  (N is smaller than Phase 1 on purpose: racecheck reports from instrumented\n"
          "   accesses, not corrupted output, and the lattice — not N — sets num_active,\n"
          "   so partial-prolog coverage is unaffected. Raise with --racecheck-n.)")
    p2_failed, p2_notes = racecheck_sweep(
        rt, index, ordered(reachable), per_op, c_in, c_out, args.racecheck_n, race_acts,
        args.timeout,
    )

    b_verdict = None
    if not args.skip_3b:
        _b_findings, b_verdict = phase_3b(
            vtod, rt, index,
            min(args.n, 50_000),  # diagnostic; it does not need the full sweep N
            c_out, args.racecheck_n, args.timeout,
        )

    print("\n" + "=" * 78)
    print("VERIFIED:")
    print(
        f"  - Phase 1 numeric: tiles {ordered(reachable)} x {len(cases)} cases "
        f"{[f'{a}/{ci}->{co}' for a, ci, co in cases]} x {args.repeats} repeats at N={args.n}"
    )
    print(
        f"  - partial-prolog path reached for "
        f"{sum(1 for v in p1_hits.values() if v is not None)}/{len(reachable)} "
        f"pool-reachable tiles"
    )
    print(
        f"  - Phase 2 racecheck: C={c_in}->{c_out}, N={args.racecheck_n}, "
        f"activities={race_acts}, 0 hazards / 0 warnings / 0 skipped required"
    )
    if args.include_unpooled and unpooled:
        print(f"  - Phase 3a numeric: pin-only tiles {ordered(unpooled)} (opt-in)")

    notes = pool_notes + p1_notes + p2_notes + p3a_notes
    if not args.include_unpooled and unpooled:
        notes.append(
            f"pin-only tiles {unpooled} were NOT run — a green Phase 1+2 is NOT validation "
            f"of them. They are pool-excluded, so nothing selects them without an explicit "
            f"pin; use --include-unpooled to cover them."
        )
    notes += [
        f"Phases 1-2 cover the POOL-REACHABLE tiles {reachable} only",
        f"racecheck ran at N={args.racecheck_n}, not Phase 1's N={args.n} (instrumentation "
        f"cost; num_active is set by the lattice, not by N)",
        f"only sm_{REQUIRED_ARCH} is covered — compile_archs=(100,) is an exact pin, so this "
        f"says nothing about any other arch",
        "this script is the residual-case check, not the primary correctness evidence — "
        "that is the author's B200 matrix",
    ]
    if args.skip_3b:
        notes.append("Phase 3b was SKIPPED — the num_k_tiles==1 / TORCH_CHECK question is open")
    print("NOT VERIFIED:")
    for note in notes:
        print(f"  - {note}")
    if b_verdict:
        print(f"PHASE 3b (advisory, C_in=32, inadmissible shape):\n  {b_verdict}")

    failures = [
        n
        for n, f in (
            ("phase 1 numeric/coverage", p1_failed),
            ("phase 1 partial-prolog coverage", p1_missed),
            ("phase 2 racecheck", p2_failed),
            ("phase 3a numeric/coverage", p3a_failed or p3a_missed),
        )
        if f
    ]
    if failures:
        print(f"\nRESULT: FAIL ({', '.join(failures)}) — tiles {TILES[0]}-{TILES[-1]} "
              f"regressed or are uncovered")
        print("=" * 78)
        return EXIT_WRONG
    print("\nRESULT: PASS — all gates and phases green for the scope listed above")
    print("=" * 78)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
