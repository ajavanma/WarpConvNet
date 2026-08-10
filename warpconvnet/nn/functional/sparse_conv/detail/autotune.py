# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Auto-tuning benchmark runners and cache management for sparse convolution
# algorithm selection.

from typing import Any, Dict, List, Optional, Tuple, Union
from jaxtyping import Float

from enum import Enum

import torch
from torch import Tensor

from warpconvnet._offset_gemm_constants import BACKEND_CUTE_GROUPED_SM90, BACKEND_CUTE_SM90
from warpconvnet.utils.benchmark_cache import (
    SpatiallySparseConvConfig,
    generic_benchmark_get_namespace,
    generic_benchmark_update_entry,
)
from warpconvnet.utils.timer import CUDATimer
from warpconvnet.utils.logger import get_logger

from .algo_params import (
    _get_filtered_AB_params,
    _get_filtered_AtB_params,
)
from .backends import (
    BwdCtx,
    FwdCtx,
    benchmark_backward,
    benchmark_forward,
    run_backward,
    run_forward,
)
from warpconvnet.constants import WARPCONVNET_AUTOTUNE_NUMERIC_CHECK

# rank_zero_only=False: auto-tuning is a per-rank, shape-dependent event, so it is
# precisely the thing that does NOT happen uniformly across ranks. Filtering to rank 0
# makes a tuning (or hanging) non-zero rank indistinguishable from one that never tuned.
logger = get_logger(__name__, rank_zero_only=False)

# ---------------------------------------------------------------------------
# Backward numeric self-check thresholds
# ---------------------------------------------------------------------------
#
# A candidate whose grad abs-sum falls below this fraction of the reference's
# abs-sum (while the reference is materially nonzero) is treated as a silent
# zero gradient — the reported cin64 dgrad failure mode.
_NUMERIC_ZERO_RATIO = 1e-3
# A candidate whose mean-relative difference from the reference exceeds this is
# treated as garbage. Generous enough that legitimate F16-accumulator tiles
# (2-3x worse rel-diff than f32, absolute ~1e-2) always pass.
_NUMERIC_RDIFF_MAX = 0.25
# Forward-only: a candidate whose WORST single element deviates from the
# reference by more than this fraction of the reference's peak magnitude is
# treated as wrong. The mean-relative bound above cannot see a defect confined
# to a small minority of rows — the GB300 tile-41 failure mode, where ~500 of
# 200k rows read up to 11x the reference. Two orders of
# magnitude looser than the suite's fp16 kernel tolerance (8e-3) so that
# accumulation-order differences never trip it.
_NUMERIC_MAX_ELEM_RDIFF = 0.5

# Benchmark iterations for auto-tuning. More iterations = more reliable
# winner selection but slower first-iteration auto-tune.
#
# Phase 1 (screening): every candidate gets _BENCHMARK_NUM_ITERS samples,
# median taken. Used to pick top-k for re-timing.
# Phase 2 (tie-break): top _BENCHMARK_TIE_BREAK_TOP_K candidates re-timed
# with _BENCHMARK_TIE_BREAK_NUM_ITERS samples, median wins.
#
# Without phase 2, top candidates within 3% of each other get ranked by
# noise — caused bimodal e2e bench at RES=32 C=1024 (best 1.24x, worst
# 2.0x same shape across runs).
_BENCHMARK_NUM_WARMUP = 3
_BENCHMARK_NUM_ITERS = 7

# Tie-break re-timing: re-time top-K candidates with more samples to
# stabilize ranking when phase-1 medians are within tie-break threshold.
_BENCHMARK_TIE_BREAK_TOP_K = 3
_BENCHMARK_TIE_BREAK_NUM_ITERS = 21
# If phase-1 best vs k-th-best ratio < this, run tie-break. Otherwise
# the gap is larger than expected noise so phase-1 ranking is trusted.
_BENCHMARK_TIE_BREAK_THRESHOLD = 1.10

# Track whether auto-tune banner has been shown (once per process)
_AUTOTUNE_BANNER_SHOWN = False
_DISTRIBUTED_WARNING_SHOWN = False


def _warn_if_distributed() -> None:
    """Warn once when auto-tuning runs inside an initialized distributed context.

    Auto-tuning is per-rank and shape-dependent: only the rank that sees a novel shape
    tunes, and while it does so it performs host-side CUDA syncs inside the autograd
    backward. Its peers meanwhile proceed to the next collective. A long tune on one rank
    therefore shows up as a collective timeout on every OTHER rank -- and because the
    tuning rank is not itself inside a collective, its own NCCL watchdog never fires, so
    the default traceback blames the innocent ranks.

    We cannot bound the tune from here without changing which kernel gets picked, but we
    can make sure the rank that is doing it says so.
    """
    global _DISTRIBUTED_WARNING_SHOWN
    if _DISTRIBUTED_WARNING_SHOWN:
        return
    try:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        rank = dist.get_rank()
        world = dist.get_world_size()
    except Exception:
        return
    _DISTRIBUTED_WARNING_SHOWN = True
    logger.warning(
        f"WarpConvNet: auto-tuning inside a distributed context (rank {rank}/{world}). "
        "Only ranks seeing a novel shape tune, and while one rank tunes its peers sit in "
        "the next collective -- a sweep that is merely SLOW under that contention can "
        "still exceed their collective timeout. Measured in the field: a backward sweep "
        "costing ~5s standalone took >600s alongside 15 ranks spinning in allreduce. "
        "This does NOT self-heal: a sweep killed by the timeout never completes, so it "
        "never caches, so the next run re-tunes and dies the same way. Pre-warm instead: "
        "python scripts/populate_benchmark_cache.py --num-voxels <your N> "
        "--channels <C_in>,<C_out> --dtypes <dtype>   (single process, no peers), or pin "
        "WARPCONVNET_{FWD,DGRAD,WGRAD}_ALGO_MODE to bound the candidate set."
    )


# ---------------------------------------------------------------------------
# In-memory benchmark result caches (config -> sorted list of results)
# ---------------------------------------------------------------------------

_BENCHMARK_AB_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[str, Dict[str, Any], float]],
] = {}  # AB gather-scatter (forward): Y = A @ B
_BENCHMARK_ABT_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[str, Dict[str, Any], float]],
] = {}  # ABt gather-scatter (dgrad): dX = dY @ W^T
_BENCHMARK_ATB_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[str, Dict[str, Any], float]],
] = {}  # AtB gather-gather (wgrad): dW = A^T @ dY

# Negative-resolution cache for a *pinned* forward algorithm filter.
#
# Maps (config, filter_key) -> (resolved_algo, resolved_params). Records that a
# pinned filter (e.g. fwd_algo="mask_gemm") had NO viable candidate for a config
# and was therefore resolved to a fallback (explicit_gemm). Without this, the
# winner cached under the config key is the fallback, which never satisfies the
# pinned filter's membership test, so every subsequent forward re-runs the full
# sweep (pinned-algo autotune thrash). The marker lets those calls short-circuit
# to the fallback with no benchmarking.
#
# Scoped to the EXACT (config, filter) identity so a different pin, a different
# algo-mode, or a different shape re-tunes instead of inheriting a stale
# fallback. Persisted in its own cache namespace (below) that is purely additive:
# it never touches the AB_gather_scatter winner records, so older cache files
# remain fully readable.
_BENCHMARK_AB_FALLBACK_RESULTS: Dict[
    Tuple[SpatiallySparseConvConfig, str],
    Tuple[str, Dict[str, Any]],
] = {}

# Cache namespace for the forward negative-resolution markers above. New in the
# v16 schema; absent from older caches (they simply behave as a first run).
_AB_FALLBACK_NAMESPACE = "AB_gather_scatter_fallback"

# Same negative-resolution idea for the BACKWARD directions, keyed by
# (namespace, config, filter_key). In-memory only: unlike the forward marker it
# is not persisted, so a pinned filter with no viable candidate re-sweeps once
# per process instead of once ever. That keeps the on-disk schema untouched.
#
# Why it exists at all: `_autotune_one_direction` used to resolve a backward
# direction purely from `cache_dict.get(cfg)`, and `cfg`
# (SpatiallySparseConvConfig) carries NO record of the algorithm filter. So on a
# warm cache -- including one loaded from disk -- `dgrad_algo=`/`wgrad_algo=`
# pins were silently ignored, and a pinned sweep's winner was written back over
# the shared adaptive winner for that shape. The forward path has guarded
# against both since the v16 schema; the backward path did not.
_BENCHMARK_BWD_FALLBACK_RESULTS: Dict[
    Tuple[str, SpatiallySparseConvConfig, str],
    Tuple[str, Dict[str, Any]],
] = {}


def _record_bwd_fallback_resolution(
    cache_ns: str,
    config: SpatiallySparseConvConfig,
    filter_key: str,
    algo: Any,
    params: Any,
) -> None:
    """Record that a pinned backward filter resolved to an out-of-filter
    fallback for ``config`` so subsequent calls short-circuit instead of
    re-sweeping every backward."""
    _BENCHMARK_BWD_FALLBACK_RESULTS[(cache_ns, config, filter_key)] = (
        _serialize_algo_value(algo),
        params if isinstance(params, dict) else {},
    )

# ---------------------------------------------------------------------------
# Serialization helpers for cache
# ---------------------------------------------------------------------------


def _serialize_algo_value(algo: Any) -> str:
    if isinstance(algo, Enum):
        return str(algo.value)
    return str(algo)


def _serialize_benchmark_results(
    results: List[Tuple[Union[str, Any], Dict[str, Any], float]],
) -> List[Tuple[str, Dict[str, Any], float]]:
    return [
        (_serialize_algo_value(algo), params, float(metric)) for algo, params, metric in results
    ]


def _algorithm_filter_key(algorithm_filter: Any) -> Optional[str]:
    """Stable string identity for a forward algorithm filter.

    Returns ``None`` for the adaptive modes (``"auto"``/``"all"``/``"trimmed"``),
    which always take the best cached winner and never consult the
    negative-resolution cache. For a pinned filter (a list of algo names) returns
    a comma-joined key so that a different pin maps to a different marker.
    """
    if isinstance(algorithm_filter, str):
        if algorithm_filter in ("auto", "all", "trimmed"):
            return None
        return algorithm_filter
    if isinstance(algorithm_filter, list):
        return ",".join(str(a) for a in algorithm_filter)
    return None


def _serialize_fallback_resolution(algo: Any, params: Any) -> List[Any]:
    """Serialize a (algo, params) resolution to a msgpack-friendly ``[str, dict]``."""
    return [_serialize_algo_value(algo), params if isinstance(params, dict) else {}]


def _deserialize_fallback_resolution(value: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse a persisted ``[algo_str, params_dict]`` back to a resolution tuple.

    Returns ``None`` for anything that does not match the expected shape so a
    corrupt or unexpected record is ignored rather than crashing the load.
    """
    if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], str):
        params = value[1] if isinstance(value[1], dict) else {}
        return (value[0], _normalize_cached_params(value[0], params))
    return None


def _record_ab_fallback_resolution(
    config: SpatiallySparseConvConfig,
    filter_key: str,
    algo: Any,
    params: Any,
) -> None:
    """Record (in memory and on disk) that a pinned forward filter resolved to a
    fallback for ``config``. Subsequent forwards with the same (config, filter)
    short-circuit to this resolution instead of re-benchmarking."""
    algo_str = _serialize_algo_value(algo)
    params_dict = params if isinstance(params, dict) else {}
    key = (config, filter_key)
    _BENCHMARK_AB_FALLBACK_RESULTS[key] = (algo_str, params_dict)
    generic_benchmark_update_entry(
        _AB_FALLBACK_NAMESPACE,
        key,
        _serialize_fallback_resolution(algo_str, params_dict),
        force=False,
    )


def _normalize_cached_params(algo: str, params: Any) -> Dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    normalized = dict(params)
    if (
        algo == "cute_implicit_gemm_sm90"
        and "tile_id" not in normalized
        and "mma_tile" in normalized
    ):
        normalized["backend"] = normalized.get("backend", BACKEND_CUTE_SM90)
        normalized["tile_id"] = normalized.pop("mma_tile")
    elif algo == "cute_grouped_sm90" and "tile_id" not in normalized and "mma_tile" in normalized:
        normalized["backend"] = normalized.get("backend", BACKEND_CUTE_GROUPED_SM90)
        normalized["tile_id"] = normalized.pop("mma_tile")
    return normalized


def _normalize_cached_algo(algo: str) -> str:
    # Cache namespaces changed with the registry migration. The algorithm names
    # stay stable; only params are rewritten by _normalize_cached_params.
    return algo


def _normalize_benchmark_results(
    results: Any,
    is_forward: bool,
) -> List[Tuple[str, Dict[str, Any], float]]:
    if results is None:
        return []
    out: List[Tuple[str, Dict[str, Any], float]] = []
    for item in results:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        algo_raw, params, metric = item
        algo_str = _normalize_cached_algo(_serialize_algo_value(algo_raw))
        out.append((algo_str, _normalize_cached_params(algo_str, params), float(metric)))
    return out


# ---------------------------------------------------------------------------
# Load cached benchmark results at module initialization
# ---------------------------------------------------------------------------


def _initialize_benchmark_cache():
    """Load cached benchmark results and populate global dictionaries."""
    ab_ns = generic_benchmark_get_namespace("AB_gather_scatter")
    abt_ns = generic_benchmark_get_namespace("ABt_gather_scatter")
    atb_ns = generic_benchmark_get_namespace("AtB_gather_gather")

    if isinstance(ab_ns, dict):
        for k, v in ab_ns.items():
            _BENCHMARK_AB_RESULTS[k] = _normalize_benchmark_results(v, is_forward=True)
    if isinstance(abt_ns, dict):
        for k, v in abt_ns.items():
            _BENCHMARK_ABT_RESULTS[k] = _normalize_benchmark_results(v, is_forward=False)
    if isinstance(atb_ns, dict):
        for k, v in atb_ns.items():
            _BENCHMARK_ATB_RESULTS[k] = _normalize_benchmark_results(v, is_forward=False)

    fb_ns = generic_benchmark_get_namespace(_AB_FALLBACK_NAMESPACE)
    if isinstance(fb_ns, dict):
        for k, v in fb_ns.items():
            parsed = _deserialize_fallback_resolution(v)
            if parsed is not None and isinstance(k, tuple) and len(k) == 2:
                _BENCHMARK_AB_FALLBACK_RESULTS[k] = parsed

    n_ab = len(ab_ns) if ab_ns else 0
    n_abt = len(abt_ns) if abt_ns else 0
    n_atb = len(atb_ns) if atb_ns else 0
    if n_ab or n_abt or n_atb:
        logger.info(
            f"Loaded {n_ab} AB_gather_scatter (fwd), {n_abt} ABt_gather_scatter (dgrad), "
            f"{n_atb} AtB_gather_gather (wgrad) benchmark configurations from cache"
        )


def _on_cache_merge(namespace: str, merged_dict: dict) -> None:
    """Callback from GenericBenchmarkCache when disk data is merged.

    Refreshes the in-memory auto-tune results with entries from other ranks.
    """
    if namespace == "AB_gather_scatter":
        for k, v in merged_dict.items():
            if k not in _BENCHMARK_AB_RESULTS:
                _BENCHMARK_AB_RESULTS[k] = _normalize_benchmark_results(v, is_forward=True)
    elif namespace == "ABt_gather_scatter":
        for k, v in merged_dict.items():
            if k not in _BENCHMARK_ABT_RESULTS:
                _BENCHMARK_ABT_RESULTS[k] = _normalize_benchmark_results(v, is_forward=False)
    elif namespace == "AtB_gather_gather":
        for k, v in merged_dict.items():
            if k not in _BENCHMARK_ATB_RESULTS:
                _BENCHMARK_ATB_RESULTS[k] = _normalize_benchmark_results(v, is_forward=False)
    elif namespace == _AB_FALLBACK_NAMESPACE:
        for k, v in merged_dict.items():
            if k not in _BENCHMARK_AB_FALLBACK_RESULTS:
                parsed = _deserialize_fallback_resolution(v)
                if parsed is not None and isinstance(k, tuple) and len(k) == 2:
                    _BENCHMARK_AB_FALLBACK_RESULTS[k] = parsed


# Initialize cache on module load
_initialize_benchmark_cache()

# Register callback so other ranks' results refresh our in-memory cache
from warpconvnet.utils.benchmark_cache import get_generic_benchmark_cache as _get_cache

_get_cache().register_on_merge_callback(_on_cache_merge)


def _tie_break_top_k(
    sorted_results: List[Tuple[str, Dict[str, Any], float]],
    run_one,
    timer,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Re-time top candidates with more samples to stabilize ranking.

    Phase-1 timed every candidate with `_BENCHMARK_NUM_ITERS` samples.
    When top candidates are close (within `_BENCHMARK_TIE_BREAK_THRESHOLD`),
    the median ranking is dominated by noise. This helper re-times the top
    `_BENCHMARK_TIE_BREAK_TOP_K` candidates with `_BENCHMARK_TIE_BREAK_NUM_ITERS`
    samples each, replaces their phase-1 medians with the new ones, and
    re-sorts.

    Args:
        sorted_results: List of (algo, params, median_ms) sorted ascending
            by median_ms.
        run_one: Callable run_one(algo, params) executing one pass; result
            ignored.
        timer: CUDA event timer with `with timer:` and `.elapsed_time`.

    Returns:
        Re-sorted result list (full length, not just top-k).
    """
    if len(sorted_results) <= 1:
        return sorted_results
    best_time = sorted_results[0][2]
    if best_time <= 0:
        return sorted_results
    # How many candidates fall within the tie-break threshold?
    cutoff = best_time * _BENCHMARK_TIE_BREAK_THRESHOLD
    in_band = [(i, r) for i, r in enumerate(sorted_results) if r[2] <= cutoff]
    in_band = in_band[:_BENCHMARK_TIE_BREAK_TOP_K]
    if len(in_band) <= 1:
        return sorted_results

    rebuilt: List[Tuple[str, Dict[str, Any], float]] = list(sorted_results)
    for i, (algo, params, _) in in_band:
        try:
            iter_times = []
            for _ in range(_BENCHMARK_TIE_BREAK_NUM_ITERS):
                with timer:
                    run_one(algo, params)
                iter_times.append(timer.elapsed_time)
            torch.cuda.synchronize()
            median = sorted(iter_times)[len(iter_times) // 2]
            rebuilt[i] = (algo, params, median)
        except (RuntimeError, Exception):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            _raise_if_context_poisoned(algo, params)
            continue
    rebuilt.sort(key=lambda x: x[2])
    return rebuilt


def _raise_if_context_poisoned(algo: str, params: Dict[str, Any]) -> None:
    """After a candidate fails, verify the CUDA context survived.

    Benign candidate failures (Python-level tile guards, unsupported-tile
    status codes, launch-config errors) are non-sticky: consuming the sync
    exception clears them and the next candidate runs normally. A device-side
    assert is STICKY — the context is dead, every subsequent launch fails, and
    the remaining candidates would all "fail" too, leaving a garbage winner
    (or none) in the cache. Convert that cascade into an immediate error that
    names the culprit candidate.
    """
    try:
        (torch.zeros(1, device="cuda") + 1).item()
    except Exception as probe_err:
        raise RuntimeError(
            f"CUDA context poisoned while benchmarking candidate {algo!r} "
            f"(params={params}): a device-side assert or other sticky error "
            f"killed the context; every subsequent kernel launch will fail. "
            f"Exclude this candidate (e.g. WARPCONVNET_*_ALGO_MODE) and "
            f"report the algo/tile."
        ) from probe_err


# ---------------------------------------------------------------------------
# Backward numeric self-check
# ---------------------------------------------------------------------------


def _fp32_reference_operands(*tensors: Optional[Tensor]) -> Optional[Tuple[Optional[Tensor], ...]]:
    """Upcast the reference operands to fp32, or ``None`` if that would OOM.

    The autotune numeric self-check used to compute its ``explicit_gemm``
    oracle **at the compute dtype**. That is inverted whenever the operands
    reach the fp16 denormal band: cuBLAS/CUTLASS flush-to-zero paths collapse
    the oracle to ~0, so the candidates that flush *agree* with it and the
    candidates that carry real signal are disqualified for "differing beyond
    tolerance".

    Measured on B200 (uniform per-batch geometry, N=500k, C=64, kv=27, fp16,
    grad_output from a ``.pow(2).mean()`` loss, i.e. |g| ~ 5e-8 which is below
    fp16's smallest normal 6.104e-5): every one of the 14 mask_gemm dgrad
    candidates was disqualified with ``zero_ratio=1.03e+04`` -- the CANDIDATE
    was ten thousand times larger than the oracle, not the other way round.
    The sweep then fell through to ``cutlass_implicit_gemm`` at 5.2599 ms
    where ``cute_grouped`` runs the same dgrad in 1.5054 ms (3.49x).

    An fp32 oracle is immune: it is computed above the denormal band, so the
    flushing candidates fail and the accurate ones pass, which is the intended
    semantics of the guard.

    LAZY as of the escalation rework: this is called only from
    ``_escalate_{fwd,bwd}_reference_to_fp32``, i.e. only on a sweep where the
    compute-dtype oracle actually produced a disqualification or carried no
    signal. Building it unconditionally cost every sweep an extra fp32
    ``explicit_gemm`` plus fp32 copies of every operand: measured on B200,
    N=500000 B=8 C=256 kv=27 fp16, peak backward CUDA memory 4232.1 MiB with the
    always-on oracle vs 3870.5 MiB with the compute-dtype oracle vs 2582.8 MiB
    with the guard off -- +361.6 MiB (+9.3%) of pure oracle, on every shape,
    almost never used.
    """
    try:
        # promote_types, not .float(): a float64 gradcheck operand must NOT be
        # downcast to fp32, or the "oracle" becomes less precise than the
        # candidates it is judging. fp16/bf16 -> fp32, fp32 -> fp32, fp64 -> fp64.
        out = tuple(
            None if t is None else t.to(torch.promote_types(t.dtype, torch.float32))
            for t in tensors
        )
        torch.cuda.synchronize()
        return out
    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:  # pragma: no cover - defensive
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        logger.debug(f"fp32 reference operands unavailable ({err}); using compute-dtype reference")
        return None


# The compute dtypes whose denormal band the compute-dtype oracle falls into.
# For fp32/fp64 compute the oracle is already exact enough and the upcast is a
# no-op, so the extra reference pass is skipped entirely.
_REDUCED_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def _reference_overflows_compute_dtype(
    ref: Optional[Tensor], compute_dtype: Optional[torch.dtype]
) -> bool:
    """True when an fp32 oracle holds values the compute dtype cannot represent.

    The pre-existing fail-open for "fp16 wgrad legitimately overflows to inf,
    so validating against that inf is meaningless" used to trigger because the
    *reference itself* was computed in fp16 and overflowed. With the fp32 oracle
    it no longer overflows, so the same situation has to be detected directly:
    if casting the oracle down to the compute dtype produces a non-finite value,
    every candidate will legitimately overflow too and the guard cannot
    discriminate. Underflow is deliberately NOT treated this way -- that is
    exactly the case the fp32 oracle exists to adjudicate.

    Cost note: the cast is monotone in magnitude, so only the PEAK element can
    overflow. Reducing first and casting the 0-d result gives the identical
    verdict for one reduction and one sync, instead of the two full-size
    temporaries (``ref.float()`` and ``rf.to(compute_dtype).float()``) the first
    implementation materialised -- 2 x N x C extra bytes on every sweep.
    """
    if ref is None or compute_dtype is None:
        return False
    if compute_dtype not in _REDUCED_PRECISION_DTYPES:
        return False
    try:
        peak = ref.detach().abs().amax()
        if not bool(torch.isfinite(peak).item()):
            return False  # handled by _reference_has_signal
        return not bool(torch.isfinite(peak.to(compute_dtype)).item())
    except Exception:  # pragma: no cover - defensive
        return False


def _reference_has_signal(ref: Optional[Tensor]) -> bool:
    """A reference grad tensor is usable for validation only if it is finite
    and carries nonzero signal.

    A non-finite reference is the crux of the wgrad false-positive: under fp16
    compute a wgrad that accumulates hundreds of thousands of coherent-sign
    terms legitimately overflows to ``inf`` (normal AMP behaviour — the
    GradScaler rescales/skips). The reference algo (explicit_gemm) overflows
    identically to every candidate, so validating candidates against that
    ``inf`` reference is meaningless. Treat it as no-signal and fail open.

    ``ref.abs()`` is reduced in fp32 rather than materialising ``ref.float()``
    first: ``amax`` propagates NaN and maps +-inf to +inf, so the finiteness
    test is unchanged, and ``sum(dtype=float32)`` accumulates above the fp16
    range without an N x C temporary. This runs once per reference AND once per
    candidate (via ``_grad_pair_disqualified``), so the temporary is not free.
    """
    if ref is None:
        return False
    ref_abs = ref.detach().abs()
    if not bool(torch.isfinite(ref_abs.amax()).item()):
        return False
    return ref_abs.sum(dtype=torch.float32).item() > 0


def _grad_pair_disqualified(
    ref: Optional[Tensor],
    cand: Optional[Tensor],
) -> Tuple[Optional[str], float, float]:
    """Compare one candidate grad tensor against its reference.

    Returns ``(reason, rdiff, zero_ratio)``. ``reason`` is ``None`` when the
    candidate is acceptable (or when there is no usable reference signal to
    check against). A non-``None`` reason means the candidate must be
    disqualified.

    Disqualification criteria (only applied when the reference carries signal,
    i.e. it is finite and ``ref.abs().sum() > 0`` -- see
    ``_reference_has_signal``):
      - candidate essentially zero: ``cand.abs().sum() / ref.abs().sum()`` is
        below ``_NUMERIC_ZERO_RATIO`` (silent zero-grad failure mode);
      - candidate non-finite while the reference is finite (NaN/Inf garbage);
      - mean-relative difference
        ``(cand - ref).abs().mean() / ref.abs().mean()`` exceeds
        ``_NUMERIC_RDIFF_MAX``.

    Comparisons run on-device; only a handful of ``.item()`` scalars are pulled
    back per pair (O(1) host syncs per candidate).
    """
    # Only validate against a finite, nonzero reference. A non-finite reference
    # (fp16 wgrad overflow) or a zero reference carries nothing to compare.
    if not _reference_has_signal(ref):
        return None, 0.0, 0.0
    ref_f = ref.float()
    ref_abs_sum = ref_f.abs().sum().item()

    if cand is None:
        return (
            "candidate produced no gradient (None) while reference is nonzero",
            float("inf"),
            0.0,
        )
    cand_f = cand.float()
    if cand_f.shape != ref_f.shape:
        return (
            f"shape mismatch cand{tuple(cand_f.shape)} vs ref{tuple(ref_f.shape)}",
            float("inf"),
            0.0,
        )

    cand_abs_sum = cand_f.abs().sum().item()
    zero_ratio = cand_abs_sum / ref_abs_sum
    if zero_ratio < _NUMERIC_ZERO_RATIO:
        return "silent zero gradient", float("nan"), zero_ratio

    # NaN/Inf garbage: reference is finite (has a real abs-sum) but the
    # candidate is not. rdiff below would be NaN and slip past the > check.
    if not torch.isfinite(cand_f).all().item():
        return "candidate contains non-finite values", float("inf"), zero_ratio

    ref_abs_mean = ref_f.abs().mean().item()
    rdiff = (cand_f - ref_f).abs().mean().item() / ref_abs_mean if ref_abs_mean > 0 else 0.0
    if rdiff > _NUMERIC_RDIFF_MAX:
        return "gradient differs from reference beyond tolerance", rdiff, zero_ratio
    return None, rdiff, zero_ratio


def _backward_numeric_disqualifies(
    ref_grads: Tuple[Optional[Tensor], Optional[Tensor]],
    cand_grads: Tuple[Optional[Tensor], Optional[Tensor]],
    needs_input_grad: Tuple[bool, ...],
) -> Optional[str]:
    """Return a disqualification reason for a backward candidate, else ``None``.

    Only the grad directions the sweep requested (``needs_input_grad``) are
    checked: grad_in for a dgrad sweep, grad_weight for a wgrad sweep.
    """
    ref_in, ref_w = ref_grads
    cand_in, cand_w = cand_grads
    checks: List[Tuple[str, Optional[Tensor], Optional[Tensor]]] = []
    if needs_input_grad[0]:
        checks.append(("grad_in", ref_in, cand_in))
    if len(needs_input_grad) > 1 and needs_input_grad[1]:
        checks.append(("grad_weight", ref_w, cand_w))
    for name, ref, cand in checks:
        reason, rdiff, zero_ratio = _grad_pair_disqualified(ref, cand)
        if reason is not None:
            return f"{name}: {reason} (rdiff={rdiff:.3g}, zero_ratio={zero_ratio:.3g})"
    return None


def _forward_numeric_disqualifies(
    ref_out: Optional[Tensor],
    cand_out: Optional[Tensor],
) -> Optional[str]:
    """Return a disqualification reason for a forward candidate, else ``None``.

    Starts from the shared backward criteria (``_grad_pair_disqualified`` is a
    generic reference-vs-candidate comparison: silent zeros, NaN/Inf, gross
    mean drift), then adds a **max-element** bound.

    The extra bound matters because the shared criteria are mean-relative, and
    the forward failures actually observed corrupt a small minority of rows: on
    GB300, ``mask_gemm`` tile 41 at C_in=32 produced up to 11x the reference
    value on ~500 of 200 000 rows. Averaged over the whole tensor that is
    invisible, so a mean-relative test passes a badly wrong kernel
    (measured on GB300).

    ``_NUMERIC_MAX_ELEM_RDIFF`` is deliberately far looser than fp16 kernel
    tolerance (the test suite uses 8e-3): the cost of disqualifying a good
    candidate is a slower winner, so the bound only needs to separate "different
    accumulation order" from "wrong".
    """
    reason, rdiff, zero_ratio = _grad_pair_disqualified(ref_out, cand_out)
    if reason is not None:
        return f"output: {reason} (rdiff={rdiff:.3g}, zero_ratio={zero_ratio:.3g})"
    if ref_out is None or cand_out is None:
        return None
    ref_f = ref_out.float()
    ref_scale = ref_f.abs().max().item()
    if ref_scale <= 0:
        return None
    max_rdiff = (cand_out.float() - ref_f).abs().max().item() / ref_scale
    if max_rdiff > _NUMERIC_MAX_ELEM_RDIFF:
        return (
            f"output: max element deviates by {max_rdiff:.3g} of the reference "
            f"peak (limit {_NUMERIC_MAX_ELEM_RDIFF})"
        )
    return None


# ---------------------------------------------------------------------------
# Forward benchmark runner
# ---------------------------------------------------------------------------


def _run_forward_benchmarks(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    warmup_iters: int = _BENCHMARK_NUM_WARMUP,
    benchmark_iters: int = _BENCHMARK_NUM_ITERS,
    custom_params: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    groups: int = 1,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Benchmark different forward algorithms and return sorted results (best first)."""
    warmup_iters = max(warmup_iters, 1)
    benchmark_iters = max(benchmark_iters, 1)

    all_benchmark_results: List[Tuple[str, Dict[str, Any], float]] = []
    timer = CUDATimer()

    def _execute_single_fwd_pass(algo_mode: str, params_config: Dict[str, Any]) -> Optional[int]:
        # Route through the shared registry so the benchmarked kernel is exactly
        # the one dispatch.py executes. Unavailable/unknown backends raise here
        # and are skipped by the surrounding try/except (same as returning a
        # non-zero status).
        ctx = FwdCtx(
            in_features=in_features,
            weight=weight,
            kernel_map=kernel_map,
            num_out_coords=num_out_coords,
            compute_dtype=compute_dtype,
            params=params_config,
            fwd_block_size=None,
            groups=groups,
        )
        return benchmark_forward(algo_mode, ctx)

    def _execute_single_fwd_capture(
        algo_mode: str, params_config: Dict[str, Any], fp32: bool = False
    ) -> Tensor:
        """Same call as ``_execute_single_fwd_pass`` but returns the output
        tensor so the numeric self-check can inspect it. Raises on a non-zero
        GEMM status.

        ``fp32=True`` runs the call on fp32 operands so the reference oracle is
        computed above the denormal band -- see ``_fp32_reference_operands``."""
        _in, _w = in_features, weight
        _cdt = compute_dtype
        if fp32:
            ops = _fp32_reference_operands(in_features, weight)
            if ops is None:
                raise RuntimeError("fp32 reference operands unavailable")
            _in, _w = ops
            _cdt = torch.float32
        ctx = FwdCtx(
            in_features=_in,
            weight=_w,
            kernel_map=kernel_map,
            num_out_coords=num_out_coords,
            compute_dtype=_cdt,
            params=params_config,
            fwd_block_size=None,
            groups=groups,
        )
        return run_forward(algo_mode, ctx)

    params_to_use = custom_params if custom_params is not None else _get_filtered_AB_params()
    # Filter out IMPLICIT_GEMM when dtype is float64 (unsupported by kernels)
    dtype_to_check = compute_dtype if compute_dtype is not None else in_features.dtype
    if dtype_to_check == torch.float64:
        params_to_use = [(algo, cfg) for (algo, cfg) in params_to_use if algo != "implicit_gemm"]

    global _AUTOTUNE_BANNER_SHOWN
    num_candidates = len(params_to_use)
    N_in = in_features.shape[0]
    C_in_val = in_features.shape[1]
    C_out_val = weight.shape[2]
    if not _AUTOTUNE_BANNER_SHOWN:
        logger.warning(
            "WarpConvNet: Auto-tuning sparse convolution algorithms. "
            "The first few iterations will be slow while optimal kernels are selected. "
            "Results are cached to ~/.cache/warpconvnet/ for future runs."
        )
        _AUTOTUNE_BANNER_SHOWN = True
    _warn_if_distributed()
    logger.info(
        f"Auto-tuning forward (N={N_in}, C_in={C_in_val}, C_out={C_out_val}, "
        f"{num_candidates} candidates)..."
    )

    # Forward numeric self-check. A wrong tile is worse than a slow one, and
    # timing alone cannot tell them apart. Mirrors the backward check, including
    # its fail-open guards — the check can NEVER force a worse winner than plain
    # timing:
    #   - reference can't be computed                          -> disabled;
    #   - reference has no usable (finite, nonzero) signal      -> escalate to an
    #     fp32 oracle; still nothing                            -> disabled;
    #   - reference fails its own check (self-inconsistent)     -> disabled;
    #   - a candidate disqualifies against the compute-dtype
    #     oracle                                                -> escalate to an
    #     fp32 oracle and re-adjudicate the whole sweep;
    #   - every runnable candidate disqualifies                 -> disabled, and
    #     the disqualified candidates are re-timed normally.
    _numeric_check_active = WARPCONVNET_AUTOTUNE_NUMERIC_CHECK
    _ref_out: Optional[Tensor] = None
    _ref_attempted = False
    _ref_is_fp32 = False
    _fp32_escalated = False
    _numeric_disabled_logged = False
    # Candidates rejected by the numeric check: (idx, algo, params). Kept so the
    # sweep can fall open and re-time them if the check rejected everything.
    _disqualified: List[Tuple[int, str, Dict[str, Any]]] = []
    # Candidates ACCEPTED against the compute-dtype oracle. Kept because an fp32
    # escalation replaces the reference mid-sweep and everything already waved
    # through has to be re-adjudicated against the new one.
    _accepted: List[Tuple[int, str, Dict[str, Any]]] = []

    def _disable_numeric_check(reason: str) -> None:
        nonlocal _numeric_check_active, _numeric_disabled_logged
        _numeric_check_active = False
        if not _numeric_disabled_logged:
            logger.warning(
                f"Auto-tune forward numeric self-check disabled for this sweep: "
                f"{reason}. Proceeding with normal timing-based selection."
            )
            _numeric_disabled_logged = True

    def _build_fwd_reference(fp32: bool) -> Optional[Tensor]:
        try:
            ref = _execute_single_fwd_capture("explicit_gemm", {}, fp32=fp32)
            torch.cuda.synchronize()
            return ref
        except Exception as err:  # pragma: no cover - defensive
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            logger.debug(f"forward reference (fp32={fp32}) failed to run ({err})")
            return None

    def _get_reference_output() -> Optional[Tensor]:
        """The CHEAP oracle: ``explicit_gemm`` at the compute dtype.

        The fp32 oracle is built lazily by ``_escalate_fwd_reference_to_fp32``
        instead of unconditionally here. It can only change a verdict where the
        compute-dtype oracle has under/overflowed, and that shows up as a
        DISQUALIFICATION (or as a signal-free reference) -- so the escalation is
        driven by the disqualification itself rather than paid for on every
        sweep. On the healthy path (no candidate disqualified, which is the
        overwhelmingly common case) the fp32 oracle is never materialised.
        """
        nonlocal _ref_out, _ref_attempted
        if _ref_attempted:
            return _ref_out
        _ref_attempted = True
        ref = _build_fwd_reference(fp32=False)
        if ref is None:
            _disable_numeric_check("reference (explicit_gemm) failed to run")
            return None
        if not _reference_has_signal(ref):
            # Total underflow or overflow-to-inf at the compute dtype: the
            # compute-dtype oracle carries nothing, so this is exactly a case
            # only the fp32 oracle can adjudicate. Escalate before failing open.
            del ref
            if _escalate_fwd_reference_to_fp32("compute-dtype reference carries no signal"):
                return _ref_out
            _disable_numeric_check(
                "reference output is non-finite or all-zero (e.g. fp16 " "accumulation overflow)"
            )
            return None
        if _forward_numeric_disqualifies(ref, ref) is not None:
            _disable_numeric_check("reference failed its own numeric check")
            return None
        _ref_out = ref
        return _ref_out

    def _escalate_fwd_reference_to_fp32(why: str) -> bool:
        """Replace the reference with an fp32 oracle. Returns True on success.

        Built at most once per sweep. Only meaningful for reduced-precision
        compute: for fp32/fp64 the "upcast" is a no-op and the two oracles are
        the same tensor.
        """
        nonlocal _ref_out, _ref_is_fp32, _fp32_escalated
        if _ref_is_fp32:
            return True
        if _fp32_escalated:
            return False
        _fp32_escalated = True
        _eff_dtype = compute_dtype if compute_dtype is not None else in_features.dtype
        if _eff_dtype not in _REDUCED_PRECISION_DTYPES:
            return False
        ref = _build_fwd_reference(fp32=True)
        if ref is None:
            return False
        if not _reference_has_signal(ref):
            return False
        if _reference_overflows_compute_dtype(ref, _eff_dtype):
            _disable_numeric_check(
                f"reference output is non-finite in {_eff_dtype}: the fp32 oracle "
                f"holds values {_eff_dtype} cannot represent, so every candidate "
                f"overflows identically"
            )
            return False
        if _forward_numeric_disqualifies(ref, ref) is not None:
            return False
        logger.info(f"Auto-tune forward: numeric guard escalated to an fp32 oracle — {why}")
        _ref_out = ref
        _ref_is_fp32 = True
        return True

    def _readjudicate_accepted_fwd() -> None:
        """Re-check everything accepted against the OLD reference.

        Without this a kernel that merely agreed with an underflowed
        compute-dtype oracle would keep its acceptance while the accurate
        kernels are judged against fp32 — i.e. exactly the inversion, kept alive
        for the candidates that ran before the escalation.
        """
        if not _accepted:
            return
        survivors: List[Tuple[int, str, Dict[str, Any]]] = []
        for idx, algo_mode, params_config in _accepted:
            try:
                cand = _execute_single_fwd_capture(algo_mode, params_config)
                torch.cuda.synchronize()
            except Exception:  # pragma: no cover - defensive
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                survivors.append((idx, algo_mode, params_config))
                continue
            reason = _forward_numeric_disqualifies(_ref_out, cand)
            del cand
            if reason is None:
                survivors.append((idx, algo_mode, params_config))
                continue
            logger.warning(
                f"Auto-tune forward: {algo_mode} {params_config} re-disqualified "
                f"against the fp32 oracle — {reason}"
            )
            _disqualified.append((idx, algo_mode, params_config))
            all_benchmark_results[:] = [
                r
                for r in all_benchmark_results
                if not (r[0] == algo_mode and r[1] == params_config)
            ]
        _accepted[:] = survivors

    def _benchmark_candidate_time(
        idx: Optional[int], algo_mode: str, params_config: Dict[str, Any]
    ) -> Optional[float]:
        """Time one already-warmed candidate; return its median ms, or ``None``
        if it fails. Poison-probes the context on failure."""
        label = f"[{idx}/{num_candidates}] " if idx is not None else ""
        iter_times = []
        try:
            for _ in range(benchmark_iters):
                with timer:
                    _execute_single_fwd_pass(algo_mode, params_config)
                iter_times.append(timer.elapsed_time)
            # Sync to catch async errors
            torch.cuda.synchronize()
        except (RuntimeError, Exception) as e:
            logger.debug(f"  {label}{algo_mode} — failed during benchmark (error: {e})")
            try:
                torch.cuda.synchronize()
            except Exception:
                pass  # Clear error state by consuming the sync exception
            _raise_if_context_poisoned(algo_mode, params_config)
            return None
        if not iter_times:
            return None
        return sorted(iter_times)[len(iter_times) // 2]

    def _record(algo_mode: str, params_config: Dict[str, Any], median_time_ms: float, idx) -> None:
        all_benchmark_results.append((algo_mode, params_config, median_time_ms))
        _param_str = ", ".join(f"{k}={v}" for k, v in params_config.items())
        label = f"[{idx}/{num_candidates}] " if idx is not None else ""
        logger.debug(
            f"  {label}{algo_mode}"
            + (f" ({_param_str})" if _param_str else "")
            + f" — {median_time_ms:.2f}ms"
        )

    for idx, (algo_mode, params_config) in enumerate(params_to_use, 1):
        # Warmup runs
        status = None
        try:
            for _ in range(warmup_iters):
                status = _execute_single_fwd_pass(algo_mode, params_config)
                if isinstance(status, int) and status != 0:
                    break
            # Sync to catch async CUDA errors from this candidate
            torch.cuda.synchronize()
        except (RuntimeError, Exception) as e:
            logger.debug(f"  [{idx}/{num_candidates}] {algo_mode} — skipped (error: {e})")
            # Clear CUDA error state to prevent corruption of subsequent candidates.
            # cudaGetLastError() resets the error flag; synchronize() then succeeds.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass  # Clear error state by consuming the sync exception
            _raise_if_context_poisoned(algo_mode, params_config)
            continue

        if isinstance(status, int) and status != 0:
            logger.debug(f"  [{idx}/{num_candidates}] {algo_mode} — skipped (unsupported)")
            continue

        # Numeric self-check, before timing: a disqualified candidate is not
        # timed at all, so it can never enter the results.
        if _numeric_check_active and algo_mode != "explicit_gemm":
            ref_out = _get_reference_output()
            if ref_out is not None:
                try:
                    cand_out = _execute_single_fwd_capture(algo_mode, params_config)
                    torch.cuda.synchronize()
                except (RuntimeError, Exception) as e:
                    logger.debug(
                        f"  [{idx}/{num_candidates}] {algo_mode} — skipped "
                        f"(numeric check run failed: {e})"
                    )
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    _raise_if_context_poisoned(algo_mode, params_config)
                    continue
                reason = _forward_numeric_disqualifies(ref_out, cand_out)
                if reason is not None and not _ref_is_fp32:
                    # The ONLY situation in which the compute-dtype oracle and
                    # an fp32 oracle can disagree, so this is where the
                    # expensive oracle earns its cost. Re-adjudicate this
                    # candidate — and everything already accepted — against it.
                    if _escalate_fwd_reference_to_fp32(f"{algo_mode} disqualified: {reason}"):
                        reason = _forward_numeric_disqualifies(_ref_out, cand_out)
                        _readjudicate_accepted_fwd()
                del cand_out
                if reason is None:
                    _accepted.append((idx, algo_mode, params_config))
                if reason is not None:
                    _param_str = ", ".join(f"{k}={v}" for k, v in params_config.items())
                    logger.warning(
                        f"Auto-tune forward: disqualifying {algo_mode}"
                        + (f" ({_param_str})" if _param_str else "")
                        + f" — {reason}"
                    )
                    _disqualified.append((idx, algo_mode, params_config))
                    continue

        median_time_ms = _benchmark_candidate_time(idx, algo_mode, params_config)
        if median_time_ms is not None:
            _record(algo_mode, params_config, median_time_ms, idx)

    # Fall open: if the check rejected every candidate it is self-evidently
    # wrong for this sweep, so re-time the rejects rather than degrade to the
    # explicit_gemm fallback.
    if not all_benchmark_results and _disqualified:
        _disable_numeric_check("every runnable candidate was disqualified")
        for idx, algo_mode, params_config in _disqualified:
            median_time_ms = _benchmark_candidate_time(idx, algo_mode, params_config)
            if median_time_ms is not None:
                _record(algo_mode, params_config, median_time_ms, idx)

    if not all_benchmark_results:
        logger.warning("No forward benchmark succeeded. Falling back to explicit_gemm.")
        with timer:
            _execute_single_fwd_pass("explicit_gemm", {})
        all_benchmark_results.append(("explicit_gemm", {}, timer.elapsed_time))

    # Sort results by time (3rd element of tuple), ascending
    all_benchmark_results.sort(key=lambda x: x[2])

    # Tie-break re-timing: when top candidates are within threshold (3% by
    # default), re-time them with more samples to stabilize ranking.
    all_benchmark_results = _tie_break_top_k(
        all_benchmark_results,
        run_one=lambda algo, cfg: _execute_single_fwd_pass(algo, cfg),
        timer=timer,
    )

    best_algo, best_params, overall_best_time_ms = all_benchmark_results[0]
    _best_param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    logger.info(
        f"Auto-tune forward complete: {best_algo}"
        + (f" ({_best_param_str})" if _best_param_str else "")
        + f" — {overall_best_time_ms:.2f}ms"
    )
    return all_benchmark_results


# ---------------------------------------------------------------------------
# Backward benchmark runner
# ---------------------------------------------------------------------------


def _run_backward_benchmarks(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    device: torch.device,
    warmup_iters: int = _BENCHMARK_NUM_WARMUP,
    benchmark_iters: int = _BENCHMARK_NUM_ITERS,
    custom_params: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    needs_input_grad: Tuple[bool, bool] = (True, True),
    groups: int = 1,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Benchmark different backward algorithms and return sorted results (best first).

    Args:
        needs_input_grad: Tuple (need_dgrad, need_wgrad). When benchmarking
            dgrad and wgrad separately, set one to False to measure only the
            other direction.
    """
    warmup_iters = max(warmup_iters, 1)
    benchmark_iters = max(benchmark_iters, 1)

    all_benchmark_results: List[Tuple[str, Dict[str, Any], float]] = []
    timer = CUDATimer()

    def _execute_single_bwd_pass(algo_mode: str, params_config: Dict[str, Any]) -> Optional[int]:
        # Route through the shared registry so the benchmarked kernel is exactly
        # the one dispatch.py executes. weight_T is None here (autotune does not
        # pre-transpose); the mask/cute_grouped backends recompute it as needed.
        ctx = BwdCtx(
            grad_output=grad_output,
            in_features=in_features,
            weight=weight,
            kernel_map=kernel_map,
            num_out_coords=num_out_coords,
            compute_dtype=compute_dtype,
            device=device,
            needs_input_grad=needs_input_grad,
            params=params_config,
            groups=groups,
        )
        return benchmark_backward(algo_mode, ctx)

    def _execute_single_bwd_capture(
        algo_mode: str, params_config: Dict[str, Any], fp32: bool = False
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        # Same ctx as the benchmarked pass, but return the raw (grad_in,
        # grad_weight) tuple so the numeric self-check can inspect the actual
        # gradients (benchmark_backward discards them, keeping only status).
        #
        # ``fp32=True`` runs on fp32 operands so the reference oracle sits above
        # the denormal band -- see ``_fp32_reference_operands``.
        _go, _in, _w = grad_output, in_features, weight
        _cdt = compute_dtype
        if fp32:
            ops = _fp32_reference_operands(grad_output, in_features, weight)
            if ops is None:
                raise RuntimeError("fp32 reference operands unavailable")
            _go, _in, _w = ops
            _cdt = torch.float32
        ctx = BwdCtx(
            grad_output=_go,
            in_features=_in,
            weight=_w,
            kernel_map=kernel_map,
            num_out_coords=num_out_coords,
            compute_dtype=_cdt,
            device=device,
            needs_input_grad=needs_input_grad,
            params=params_config,
            groups=groups,
        )
        return run_backward(algo_mode, ctx)

    # Numeric self-check state. The explicit_gemm reference is computed once per
    # sweep on the same probe tensors, reused for every candidate, and never
    # persisted. Several fail-open guards ensure the check can NEVER force a
    # worse winner than plain timing:
    #   - reference can't be computed                              -> disabled;
    #   - reference has no usable (finite, nonzero) signal in any
    #     requested direction (e.g. fp16 wgrad overflow -> inf)    -> escalate to
    #     an fp32 oracle; if that one holds values the compute dtype cannot
    #     represent (the real wgrad overflow)                      -> disabled;
    #   - a candidate disqualifies against the compute-dtype oracle -> escalate
    #     to an fp32 oracle and re-adjudicate the whole sweep;
    #   - reference fails its own check (self-inconsistent)        -> disabled;
    #   - every runnable candidate disqualifies (check is
    #     self-evidently invalid for this sweep)                   -> disabled,
    #     and the disqualified candidates are re-timed normally.
    _numeric_check_active = WARPCONVNET_AUTOTUNE_NUMERIC_CHECK
    _ref_grads: Optional[Tuple[Optional[Tensor], Optional[Tensor]]] = None
    _ref_attempted = False
    _ref_is_fp32 = False
    _fp32_escalated = False
    _numeric_disabled_logged = False
    # Candidates rejected by the numeric check: (idx, algo, params). Kept so the
    # sweep can fall open and re-time them if the check rejected everything.
    _disqualified: List[Tuple[int, str, Dict[str, Any]]] = []
    # Candidates ACCEPTED against the compute-dtype oracle -- re-adjudicated if
    # an fp32 escalation replaces the reference mid-sweep.
    _accepted: List[Tuple[int, str, Dict[str, Any]]] = []

    def _disable_numeric_check(reason: str) -> None:
        nonlocal _numeric_check_active, _numeric_disabled_logged
        _numeric_check_active = False
        if not _numeric_disabled_logged:
            logger.warning(
                f"Auto-tune backward numeric self-check disabled for this sweep: "
                f"{reason}. Proceeding with normal timing-based selection."
            )
            _numeric_disabled_logged = True

    def _build_bwd_reference(
        fp32: bool,
    ) -> Optional[Tuple[Optional[Tensor], Optional[Tensor]]]:
        try:
            ref = _execute_single_bwd_capture("explicit_gemm", {}, fp32=fp32)
            torch.cuda.synchronize()
            return ref
        except Exception as err:  # pragma: no cover - defensive
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            logger.debug(f"backward reference (fp32={fp32}) failed to run ({err})")
            return None

    def _bwd_reference_usable(ref) -> bool:
        ref_in, ref_w = ref
        return (needs_input_grad[0] and _reference_has_signal(ref_in)) or (
            len(needs_input_grad) > 1 and needs_input_grad[1] and _reference_has_signal(ref_w)
        )

    def _get_reference_grads() -> Optional[Tuple[Optional[Tensor], Optional[Tensor]]]:
        """The CHEAP oracle: ``explicit_gemm`` at the compute dtype.

        See ``_run_forward_benchmarks._get_reference_output`` — the fp32 oracle
        is built only when a verdict actually depends on it.
        """
        nonlocal _ref_grads, _ref_attempted
        if _ref_attempted:
            return _ref_grads
        _ref_attempted = True
        ref = _build_bwd_reference(fp32=False)
        if ref is None:
            _disable_numeric_check("reference (explicit_gemm) failed to run")
            _ref_grads = None
            return None
        # The reference must carry usable (finite, nonzero) signal in at least
        # one requested direction. A non-finite reference — the fp16 wgrad
        # overflow case, where the reference algo overflows to inf identically
        # to every candidate — cannot validate anything; a totally underflowed
        # one is the case the fp32 oracle exists for, so escalate first.
        if not _bwd_reference_usable(ref):
            del ref
            if _escalate_bwd_reference_to_fp32("compute-dtype reference carries no signal"):
                return _ref_grads
            _disable_numeric_check(
                "reference gradient is non-finite or all-zero in every checked "
                "direction (e.g. fp16 accumulation overflow)"
            )
            _ref_grads = None
            return None
        # Sanity: the reference must pass its own check. If it doesn't, the
        # criteria are self-inconsistent here and cannot be trusted.
        if _backward_numeric_disqualifies(ref, ref, needs_input_grad) is not None:
            _disable_numeric_check("reference failed its own numeric check")
            _ref_grads = None
            return None
        _ref_grads = ref
        return _ref_grads

    def _escalate_bwd_reference_to_fp32(why: str) -> bool:
        """Replace the reference gradients with an fp32 oracle. Built at most
        once per sweep; see ``_run_forward_benchmarks._escalate_fwd_reference_to_fp32``."""
        nonlocal _ref_grads, _ref_is_fp32, _fp32_escalated
        if _ref_is_fp32:
            return True
        if _fp32_escalated:
            return False
        _fp32_escalated = True
        _eff_dtype = compute_dtype if compute_dtype is not None else grad_output.dtype
        if _eff_dtype not in _REDUCED_PRECISION_DTYPES:
            return False
        ref = _build_bwd_reference(fp32=True)
        if ref is None:
            return False
        if not _bwd_reference_usable(ref):
            return False
        ref_in, ref_w = ref
        if _reference_overflows_compute_dtype(ref_in, _eff_dtype) or (
            _reference_overflows_compute_dtype(ref_w, _eff_dtype)
        ):
            _disable_numeric_check(
                f"reference gradient is non-finite in {_eff_dtype}: the fp32 "
                f"oracle holds values {_eff_dtype} cannot represent, so every "
                f"candidate overflows identically"
            )
            return False
        if _backward_numeric_disqualifies(ref, ref, needs_input_grad) is not None:
            return False
        logger.info(f"Auto-tune backward: numeric guard escalated to an fp32 oracle — {why}")
        _ref_grads = ref
        _ref_is_fp32 = True
        return True

    def _readjudicate_accepted_bwd() -> None:
        """Re-check everything accepted against the OLD (compute-dtype) oracle."""
        if not _accepted:
            return
        survivors: List[Tuple[int, str, Dict[str, Any]]] = []
        for idx, algo_mode, params_config in _accepted:
            try:
                cand = _execute_single_bwd_capture(algo_mode, params_config)
                torch.cuda.synchronize()
            except Exception:  # pragma: no cover - defensive
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                survivors.append((idx, algo_mode, params_config))
                continue
            reason = _backward_numeric_disqualifies(_ref_grads, cand, needs_input_grad)
            del cand
            if reason is None:
                survivors.append((idx, algo_mode, params_config))
                continue
            logger.warning(
                f"Auto-tune backward: {algo_mode} {params_config} re-disqualified "
                f"against the fp32 oracle — {reason}"
            )
            _disqualified.append((idx, algo_mode, params_config))
            all_benchmark_results[:] = [
                r
                for r in all_benchmark_results
                if not (r[0] == algo_mode and r[1] == params_config)
            ]
        _accepted[:] = survivors

    def _benchmark_candidate_time(
        idx: Optional[int], algo_mode: str, params_config: Dict[str, Any]
    ) -> Optional[float]:
        """Warm up and time one candidate; return its median ms, or ``None`` if
        it is unsupported or fails. Poison-probes the context on failure."""
        label = f"[{idx}/{num_candidates}] " if idx is not None else ""
        status = None
        try:
            for _ in range(warmup_iters):
                status = _execute_single_bwd_pass(algo_mode, params_config)
                if isinstance(status, int) and status != 0:
                    break
            torch.cuda.synchronize()
        except (RuntimeError, Exception) as e:
            logger.debug(f"  {label}{algo_mode} — skipped (error: {e})")
            try:
                torch.cuda.synchronize()
            except Exception:
                pass  # Clear error state by consuming the sync exception
            _raise_if_context_poisoned(algo_mode, params_config)
            return None

        if isinstance(status, int) and status != 0:
            logger.debug(f"  {label}{algo_mode} — skipped (unsupported)")
            return None

        iter_times = []
        if benchmark_iters == 0:
            if warmup_iters == 0:
                return None
        else:
            try:
                for _ in range(benchmark_iters):
                    with timer:
                        _execute_single_bwd_pass(algo_mode, params_config)
                    iter_times.append(timer.elapsed_time)
                torch.cuda.synchronize()
            except (RuntimeError, Exception) as e:
                logger.debug(f"  {label}{algo_mode} — failed during benchmark (error: {e})")
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass  # Clear error state by consuming the sync exception
                _raise_if_context_poisoned(algo_mode, params_config)
                return None

        if iter_times:
            median_time_ms = sorted(iter_times)[len(iter_times) // 2]
            _param_str = ", ".join(f"{k}={v}" for k, v in params_config.items())
            logger.debug(
                f"  {label}{algo_mode}"
                + (f" ({_param_str})" if _param_str else "")
                + f" — {median_time_ms:.2f}ms"
            )
            return median_time_ms
        return None

    params_to_use = custom_params if custom_params is not None else _get_filtered_AtB_params()
    # Filter out IMPLICIT_GEMM when dtype is float64 (unsupported by kernels)
    dtype_to_check = compute_dtype if compute_dtype is not None else grad_output.dtype
    if dtype_to_check == torch.float64:
        params_to_use = [(algo, cfg) for (algo, cfg) in params_to_use if algo != "implicit_gemm"]

    global _AUTOTUNE_BANNER_SHOWN
    num_candidates = len(params_to_use)
    N_in = in_features.shape[0]
    C_in_val = in_features.shape[1]
    C_out_val = weight.shape[2]
    if not _AUTOTUNE_BANNER_SHOWN:
        logger.warning(
            "WarpConvNet: Auto-tuning sparse convolution algorithms. "
            "The first few iterations will be slow while optimal kernels are selected. "
            "Results are cached to ~/.cache/warpconvnet/ for future runs."
        )
        _AUTOTUNE_BANNER_SHOWN = True
    _warn_if_distributed()
    logger.info(
        f"Auto-tuning backward (N={N_in}, C_in={C_in_val}, C_out={C_out_val}, "
        f"{num_candidates} candidates)..."
    )

    for idx, (algo_mode, params_config) in enumerate(params_to_use, 1):
        # Numeric self-check: verify the candidate produces a numerically
        # correct gradient before timing it. A candidate that silently returns
        # a zero (or garbage) gradient is disqualified exactly like one that
        # raised — excluded from selection, never cached as a winner. The
        # capture also serves as a warmup run. See _backward_numeric_disqualifies.
        if _numeric_check_active:
            ref_grads = _get_reference_grads()
            if ref_grads is not None:
                try:
                    cand_grads = _execute_single_bwd_capture(algo_mode, params_config)
                    torch.cuda.synchronize()
                except (RuntimeError, Exception) as e:
                    logger.debug(
                        f"  [{idx}/{num_candidates}] {algo_mode} — "
                        f"skipped (numeric-check capture error: {e})"
                    )
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    _raise_if_context_poisoned(algo_mode, params_config)
                    continue
                reason = _backward_numeric_disqualifies(ref_grads, cand_grads, needs_input_grad)
                if reason is not None and not _ref_is_fp32:
                    # A disqualification is the only verdict an fp32 oracle can
                    # overturn, so build it here instead of on every sweep.
                    if _escalate_bwd_reference_to_fp32(f"{algo_mode} disqualified: {reason}"):
                        reason = _backward_numeric_disqualifies(
                            _ref_grads, cand_grads, needs_input_grad
                        )
                        _readjudicate_accepted_bwd()
                if reason is None:
                    _accepted.append((idx, algo_mode, params_config))
                if reason is not None:
                    _param_str = ", ".join(f"{k}={v}" for k, v in params_config.items())
                    logger.warning(
                        f"  [{idx}/{num_candidates}] {algo_mode}"
                        + (f" ({_param_str})" if _param_str else "")
                        + f" — DISQUALIFIED by numeric self-check: {reason}"
                    )
                    _disqualified.append((idx, algo_mode, params_config))
                    continue

        median_time_ms = _benchmark_candidate_time(idx, algo_mode, params_config)
        if median_time_ms is not None:
            all_benchmark_results.append((algo_mode, params_config, median_time_ms))

    # Fail-open: if the numeric check rejected EVERY runnable candidate, it is
    # self-evidently invalid for this sweep (a healthy sweep always has at least
    # the correct reference-class winner). Disable it and re-time the rejected
    # candidates so selection proceeds normally instead of collapsing to the
    # slow explicit_gemm fallback below.
    if not all_benchmark_results and _disqualified:
        _disable_numeric_check(f"all {len(_disqualified)} runnable candidate(s) were disqualified")
        for idx, algo_mode, params_config in _disqualified:
            median_time_ms = _benchmark_candidate_time(idx, algo_mode, params_config)
            if median_time_ms is not None:
                all_benchmark_results.append((algo_mode, params_config, median_time_ms))

    if not all_benchmark_results:
        logger.warning("No backward benchmark succeeded. Falling back to explicit_gemm.")
        with timer:
            _execute_single_bwd_pass("explicit_gemm", {})
        all_benchmark_results.append(("explicit_gemm", {}, timer.elapsed_time))

    # Sort results by time (3rd element of tuple), ascending
    all_benchmark_results.sort(key=lambda x: x[2])

    # Tie-break re-timing for stable ranking when top candidates are
    # within threshold. See _tie_break_top_k docstring.
    all_benchmark_results = _tie_break_top_k(
        all_benchmark_results,
        run_one=lambda algo, cfg: _execute_single_bwd_pass(algo, cfg),
        timer=timer,
    )

    best_algo, best_params, overall_best_time_ms = all_benchmark_results[0]
    _best_param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    logger.info(
        f"Auto-tune backward complete: {best_algo}"
        + (f" ({_best_param_str})" if _best_param_str else "")
        + f" — {overall_best_time_ms:.2f}ms"
    )
    return all_benchmark_results
