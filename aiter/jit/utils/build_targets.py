# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Pure-Python arch constants and env-driven build target resolution.
# No torch dependency — safe to import in build scripts, gen_instances, and tests
# that run without a GPU or a full PyTorch install.
import os
import sys

# core.py imports this file as `build_targets` via sys.path; mha.py and tests
# import `aiter.jit.utils.build_targets`. One module object keeps
# UnmappedCkFmhaTargetsError identity consistent for except / pytest.raises.
sys.modules.setdefault("build_targets", sys.modules[__name__])
sys.modules.setdefault("aiter.jit.utils.build_targets", sys.modules[__name__])

GFX_MAP = {
    0: "native",
    1: "gfx90a",
    2: "gfx908",
    3: "gfx940",
    4: "gfx941",
    5: "gfx942",
    6: "gfx945",
    7: "gfx1100",
    8: "gfx950",
    9: "gfx1101",
    10: "gfx1102",
    11: "gfx1103",
    12: "gfx1150",
    13: "gfx1151",
    14: "gfx1152",
    15: "gfx1153",
    16: "gfx1200",
    17: "gfx1201",
    18: "gfx1250",
}

# Maps gfx arch to the default (SPX / full-GPU) CU count used when no live GPU is
# present at build time (e.g. CI nodes with GPU_ARCHS set but no device visible).
# For live GPU builds, get_cu_num() is used instead and correctly reflects the
# actual visible CU count, including non-SPX partition modes (DPX / QPX / CPX)
# and binned variants (e.g. MI308X is gfx942 but has fewer CUs than MI300X).
# If building without a GPU for a binned or partitioned target, set CU_NUM
# explicitly alongside GPU_ARCHS to override the default here.
# Extend this table when adding support for new GPU targets.
GFX_CU_NUM_MAP = {
    "gfx942": 304,  # MI300X (SPX, full GPU); MI308X shares gfx942 — use CU_NUM override
    "gfx950": 256,  # MI350
    "gfx1250": 256,  # Gfx1250
}


def _parse_gpu_archs_env(gfx_env: str) -> list[str]:
    """Split a GPU_ARCHS string into a list of non-empty architecture names.

    Raises RuntimeError if no valid architecture names remain after splitting
    on ';' and stripping whitespace — e.g. GPU_ARCHS=" ; " would otherwise
    silently produce an empty target list and fall back to heuristic kernels.
    """
    archs = [g.strip() for g in gfx_env.split(";") if g.strip()]
    if not archs:
        raise RuntimeError(
            f"GPU_ARCHS={gfx_env!r} contains no valid architecture names after splitting on ';'. "
            f"Known targets: {list(GFX_CU_NUM_MAP.keys())}"
        )
    return archs


def get_build_targets_env() -> list[tuple[str, int]]:
    """Resolve build targets from GPU_ARCHS env var only.  No live GPU detection.

    Raises RuntimeError if GPU_ARCHS is not set or contains an unknown arch.
    Intended for CI nodes, build scripts, and tests that run without a GPU.
    Use chip_info.get_build_targets() when live GPU fallback is also desired.
    """
    gfx_env = os.getenv("GPU_ARCHS")
    if not gfx_env:
        raise RuntimeError(
            "GPU_ARCHS is not set. "
            "Set GPU_ARCHS=gfx942 (or similar) to resolve build targets without a GPU."
        )
    targets = []
    for gfx in _parse_gpu_archs_env(gfx_env):
        if gfx not in GFX_CU_NUM_MAP:
            raise RuntimeError(
                f"Unknown gfx '{gfx}' in GPU_ARCHS — add it to "
                f"GFX_CU_NUM_MAP in build_targets.py. Known targets: "
                f"{list(GFX_CU_NUM_MAP.keys())}"
            )
        cu_num = int(os.getenv("CU_NUM", GFX_CU_NUM_MAP[gfx]))
        targets.append((gfx, cu_num))
    return targets


def filter_tune_df(tune_df, targets: list):
    """Return the subset of tune_df whose (gfx, cu_num) matches any entry in targets.

    Args:
        tune_df:  pandas DataFrame loaded from a tuning CSV (must have 'gfx' and
                  'cu_num' columns).
        targets:  list of (gfx, cu_num) tuples, as returned by get_build_targets()
                  or get_build_targets_env().

    Returns:
        Filtered DataFrame (original index preserved, no reset).
    """
    import pandas as pd

    mask = pd.Series([False] * len(tune_df), index=tune_df.index)
    for gfx, cu_num in targets:
        mask |= (tune_df["gfx"] == gfx) & (tune_df["cu_num"] == cu_num)
    return tune_df[mask]


# CK 01_fmha/generate.py --targets factory keys. Longest prefix first so
# gfx950 is not gfx9, gfx115 is not gfx11, gfx1250 is not gfx12.
_CK_FMHA_ARCH_PREFIXES: tuple[tuple[str, str], ...] = (
    ("gfx950", "gfx950"),
    ("gfx125", "gfx125"),
    ("gfx120", "gfx12"),
    ("gfx115", "gfx115"),
    ("gfx11", "gfx11"),
    ("gfx9", "gfx9"),
)


class UnmappedCkFmhaTargetsError(RuntimeError):
    """No GPU_ARCHS entry maps to a CK fmha factory key.

    ck_fmha_targets() raises this instead of falling back to CK's default
    gfx9,gfx950. get_args_of_build("all") catches it per module so an
    unmapped arch (cpu, gfx1030, ...) does not abort unrelated prebuilds
    (this guard covers the get_args_of_build("all") traversal used by
    PREBUILD_KERNELS 2/3; the PREBUILD_KERNELS==1 path in setup.py calls
    the mha-specific builders directly and still raises on an unmapped arch).
    """


def map_gpu_archs_to_ck_fmha_targets(archs: list[str]) -> list[str]:
    """Map GPU_ARCHS / get_gfx_list() names to CK fmha factory keys.

    Unknown names are omitted. Already-emitted keys are skipped. Order follows
    the input arch list.
    """
    keys: list[str] = []
    seen: set[str] = set()
    for arch in archs:
        name = arch.split(":", 1)[0].lower()
        mapped = None
        for prefix, key in _CK_FMHA_ARCH_PREFIXES:
            if name.startswith(prefix):
                mapped = key
                break
        if mapped is None or mapped in seen:
            continue
        seen.add(mapped)
        keys.append(mapped)
    return keys


def ck_fmha_targets() -> str:
    """Comma-joined CK fmha --targets for the current get_gfx_list().

    Raises UnmappedCkFmhaTargetsError (a RuntimeError subclass) if nothing
    maps — never fall back to CK's default gfx9,gfx950.
    """
    from chip_info import get_gfx_list  # lazy: chip_info imports this module

    archs = get_gfx_list()
    keys = map_gpu_archs_to_ck_fmha_targets(archs)
    if not keys:
        raise UnmappedCkFmhaTargetsError(
            f"No CK fmha --targets mapping for GPU_ARCHS={archs!r}; "
            "refusing to fall back to CK default gfx9,gfx950"
        )
    return ",".join(keys)


def ck_fmha_batch_prefill_targets() -> str:
    """gfx9* subset of ck_fmha_targets() keys. Empty string if none (do not raise)."""
    from chip_info import get_gfx_list  # lazy: chip_info imports this module

    keys = [
        k
        for k in map_gpu_archs_to_ck_fmha_targets(get_gfx_list())
        if k.startswith("gfx9")
    ]
    return ",".join(keys)


def ck_fmha_batch_prefill_gen_targets() -> str:
    """--targets for 01_fmha generate.py -d batch_prefill.

    CK drops *all* batch_prefill kernels if any --targets token is not gfx9*
    (has_non_gfx9 in example/ck_tile/01_fmha/codegen/ops/fmha_batch_prefill.py).
    Pass the gfx9* subset when the mapped list has one; otherwise pass
    ck_fmha_targets() (e.g. gfx11) so generate.py still emits
    fmha_batch_prefill_api.cpp.

    module_mha_batch_prefill and libmha_fwd both compile
    cpp_itfs/mha_fwd_batch_prefill.cu, which calls fmha_batch_prefill().
    Skipping generate.py (empty blob_gen_cmd) only skips staging sources —
    PREBUILD=2 still builds the module and then link-fails. Do not return ''.
    """
    return ck_fmha_batch_prefill_targets() or ck_fmha_targets()
