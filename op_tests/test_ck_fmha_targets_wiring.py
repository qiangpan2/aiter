# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""JSON / cmdGenFunc / compile.py / runtime-gate wiring for CK fmha --targets."""

import importlib.util
import os
from unittest.mock import patch

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _target_flag(cmd: str) -> str | None:
    if "--targets" not in cmd:
        return None
    after = cmd.split("--targets", 1)[1].strip()
    return after.split()[0]


def _dummy_qkv(dtype):
    q = torch.empty(1, 8, 2, 64, dtype=dtype)
    k = torch.empty(1, 8, 2, 64, dtype=dtype)
    v = torch.empty(1, 8, 2, 64, dtype=dtype)
    return q, k, v


def _load_cpp_mha_compile():
    path = os.path.join(_REPO_ROOT, "op_tests", "cpp", "mha", "compile.py")
    spec = importlib.util.spec_from_file_location("cpp_mha_compile", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_core_exports_helpers_for_json_eval():
    from aiter.jit import core as jit_core

    assert callable(jit_core.ck_fmha_targets)
    assert callable(jit_core.ck_fmha_batch_prefill_gen_targets)
    assert not hasattr(jit_core, "ck_fmha_batch_prefill_targets")


def test_json_fwd_bwd_use_ck_fmha_targets():
    from aiter.jit import core as jit_core

    with (
        patch.object(jit_core, "ck_fmha_targets", lambda: "SENTINEL_FWD"),
        patch.object(
            jit_core, "ck_fmha_batch_prefill_gen_targets", lambda: "SENTINEL_BP"
        ),
    ):
        for name in (
            "module_mha_fwd",
            "module_mha_varlen_fwd",
            "module_mha_bwd",
            "module_mha_varlen_bwd",
            "libmha_fwd",
            "libmha_bwd",
        ):
            cmds = jit_core.get_args_of_build(name)["blob_gen_cmd"]
            if isinstance(cmds, str):
                cmds = [cmds]
            gen_cmds = [
                c
                for c in cmds
                if "01_fmha/generate.py" in c and "batch_prefill" not in c
            ]
            assert gen_cmds, f"{name} missing 01_fmha generate.py (non-batch_prefill)"
            for cmd in gen_cmds:
                assert _target_flag(cmd) == "SENTINEL_FWD", cmd
                assert "gfx1100" not in cmd
            for cmd in cmds:
                if "hsa/codegen.py" in cmd:
                    assert "--targets" not in cmd, cmd


def test_json_hsa_v3_and_sparse_untouched():
    from aiter.jit import core as jit_core

    for name in (
        "module_fmha_v3_bwd",
        "module_fmha_v3_varlen_bwd",
        "module_fmha_v3_varlen_fwd",
        "module_vsa_sparse_attention",
    ):
        cmds = jit_core.get_args_of_build(name)["blob_gen_cmd"]
        if isinstance(cmds, str):
            cmds = [cmds]
        for cmd in cmds:
            assert "--targets" not in cmd, (name, cmd)


def test_json_batch_prefill_uses_named_helper():
    """PREBUILD still compiles module_mha_batch_prefill (srcs include
    mha_fwd_batch_prefill.cu → fmha_batch_prefill()). json must call
    ck_fmha_batch_prefill_gen_targets so RDNA gets an api-only stub, not ''.
    """
    from aiter.jit import core as jit_core

    with (
        patch.object(
            jit_core, "ck_fmha_batch_prefill_gen_targets", lambda: "SENTINEL_GEN"
        ),
        patch.object(jit_core, "ck_fmha_targets", lambda: "SENTINEL_FWD"),
    ):
        for name in ("module_mha_batch_prefill", "libmha_fwd"):
            cmds = jit_core.get_args_of_build(name)["blob_gen_cmd"]
            if isinstance(cmds, str):
                cmds = [cmds]
            bp = next(c for c in cmds if "batch_prefill" in c)
            assert _target_flag(bp) == "SENTINEL_GEN", bp


@pytest.mark.parametrize(
    "gfx_list, fwd, bp",
    [
        (["gfx942", "gfx1100"], "gfx9,gfx11", "gfx9"),
        (["gfx1100"], "gfx11", "gfx11"),
        (["gfx950"], "gfx950", "gfx950"),
    ],
)
def test_json_helpers_follow_get_gfx_list(gfx_list, fwd, bp):
    """json eval → real helpers (no sentinel). mixed gfx9*+gfx11 keeps
    batch_prefill on the gfx9* subset; pure RDNA uses fwd keys (api-only)."""
    from aiter.jit import core as jit_core

    with patch("chip_info.get_gfx_list", return_value=gfx_list):
        bp_cmds = jit_core.get_args_of_build("module_mha_batch_prefill")["blob_gen_cmd"]
        if isinstance(bp_cmds, str):
            bp_cmds = [bp_cmds]
        assert _target_flag(bp_cmds[0]) == bp, bp_cmds[0]

        lib = jit_core.get_args_of_build("libmha_fwd")["blob_gen_cmd"]
        if isinstance(lib, str):
            lib = [lib]
        gen = [c for c in lib if "01_fmha/generate.py" in c]
        bp_lib = [c for c in gen if "batch_prefill" in c]
        fwd_lib = [c for c in gen if "batch_prefill" not in c]
        assert bp_lib and all(_target_flag(c) == bp for c in bp_lib), bp_lib
        assert fwd_lib and all(_target_flag(c) == fwd for c in fwd_lib), fwd_lib


def test_cmdGenFunc_fwd_bwd_pass_factory_targets():
    from aiter.ops.mha import cmdGenFunc_mha_bwd, cmdGenFunc_mha_fwd

    q, k, v = _dummy_qkv(torch.float16)
    with patch("aiter.ops.mha.ck_fmha_targets", lambda: "SENTINEL_FWD"):
        fwd = cmdGenFunc_mha_fwd(q, k, v, 0.0, 0.1, False, -1, -1, 0, False, False)[
            "blob_gen_cmd"
        ]
        assert _target_flag(fwd[0]) == "SENTINEL_FWD"
        assert "--receipt 100" in fwd[0]

        dout = torch.empty_like(q)
        out = torch.empty_like(q)
        lse = torch.empty(1, 2, 8, dtype=torch.float32)
        bwd = cmdGenFunc_mha_bwd(
            dout, q, k, v, out, lse, 0.0, 0.1, False, -1, -1, True
        )["blob_gen_cmd"]
        assert _target_flag(bwd[0]) == "SENTINEL_FWD"
        assert "hsa/codegen.py" in bwd[1]
        assert "--targets" not in bwd[1]


def test_mha_recipes_no_device_name_targets():
    from aiter.jit.utils import mha_recipes
    from aiter.jit.utils.mha_recipes import get_mha_varlen_prebuild_variants_by_names

    assert not hasattr(mha_recipes, "_ck_targets_flag")
    with patch.object(mha_recipes, "ck_fmha_targets", lambda: "SENTINEL_FWD"):
        variants = get_mha_varlen_prebuild_variants_by_names(
            ["mha_varlen_fwd_fp16_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale"],
            "/ck",
        )
        for cmd in variants[0]["blob_gen_cmd"]:
            assert _target_flag(cmd) == "SENTINEL_FWD"
            assert "gfx1100" not in cmd


def test_cpp_compile_py_targets():
    mod = _load_cpp_mha_compile()
    with (
        patch.object(mod, "ck_fmha_targets", lambda: "SENTINEL_FWD"),
        patch.object(mod, "ck_fmha_batch_prefill_gen_targets", lambda: "SENTINEL_BP"),
    ):
        cmds = mod.cmdGenFunc_mha_fwd(False)["blob_gen_cmd"]
        gen = [
            c for c in cmds if "01_fmha/generate.py" in c and "batch_prefill" not in c
        ]
        assert gen and all(_target_flag(c) == "SENTINEL_FWD" for c in gen)
        hsa = [c for c in cmds if "hsa/codegen.py" in c]
        assert hsa and all("--targets" not in c for c in hsa)
        assert (
            _target_flag(next(c for c in cmds if "batch_prefill" in c)) == "SENTINEL_BP"
        )
        bwd = mod.cmdGenFunc_mha_bwd(False)["blob_gen_cmd"]
        assert _target_flag(bwd[0]) == "SENTINEL_FWD"
        assert "--targets" not in bwd[1]


def test_cmdGenFunc_batch_prefill_uses_gen_targets():
    from aiter.ops.mha import cmdGenFunc_mha_batch_prefill

    q = torch.empty(8, 2, 64, dtype=torch.float16)
    k = torch.empty(4, 16, 2, 64, dtype=torch.float16)
    v = torch.empty(4, 16, 2, 64, dtype=torch.float16)
    cu = torch.zeros(2, dtype=torch.int32)
    with (
        patch("aiter.ops.mha.ck_fmha_batch_prefill_gen_targets", lambda: "gfx9"),
        patch("aiter.ops.mha.get_gfx", lambda: "gfx942"),
        patch("aiter.ops.mha.map_gpu_archs_to_ck_fmha_targets", lambda archs: ["gfx9"]),
    ):
        cmds = cmdGenFunc_mha_batch_prefill(
            q,
            k,
            v,
            cu,
            cu,
            cu,
            8,
            8,
            0.0,
            0.1,
            0.0,
            False,
            False,
            -1,
            -1,
            0,
            False,
            False,
        )["blob_gen_cmd"]
        assert _target_flag(cmds[0]) == "gfx9"
        assert "--receipt 200" in cmds[0]


@pytest.mark.parametrize(
    "arch, should_raise",
    [
        ("gfx1100", True),
        ("gfx1151", True),
        ("gfx1201", False),
        ("gfx942", False),
    ],
)
def test_fp8_cmdGenFunc_gate(arch, should_raise):
    from aiter import dtypes
    from aiter.ops.mha import cmdGenFunc_mha_fwd

    q, k, v = _dummy_qkv(dtypes.fp8)
    gfx_patch = patch("aiter.ops.mha.get_gfx", return_value=arch)
    if should_raise:
        with gfx_patch, pytest.raises(NotImplementedError, match=arch) as exc:
            cmdGenFunc_mha_fwd(q, k, v, 0.0, 0.1, False, -1, -1, 0, False, False)
        assert "fp8" in str(exc.value).lower() or "FP8" in str(exc.value)
        return
    key = {"gfx1201": "gfx12", "gfx942": "gfx9"}[arch]
    with gfx_patch, patch("aiter.ops.mha.ck_fmha_targets", lambda: key):
        out = cmdGenFunc_mha_fwd(q, k, v, 0.0, 0.1, False, -1, -1, 0, False, False)
        assert "blob_gen_cmd" in out


def test_fp8_out_cmdGenFunc_not_implemented():
    from aiter import dtypes
    from aiter.ops.mha import cmdGenFunc_mha_fwd

    q, k, v = _dummy_qkv(dtypes.fp8)
    out = torch.empty_like(q)
    with (
        patch("aiter.ops.mha.get_gfx", return_value="gfx1201"),
        pytest.raises(NotImplementedError, match="Unsupported output dtype"),
    ):
        cmdGenFunc_mha_fwd(q, k, v, 0.0, 0.1, False, -1, -1, 0, False, False, out=out)


def test_non_gfx9_batch_prefill_raises_before_compile():
    from aiter.ops.mha import cmdGenFunc_mha_batch_prefill, mha_batch_prefill_func

    with patch("aiter.ops.mha.get_gfx", return_value="gfx1100"):
        with pytest.raises(NotImplementedError, match="gfx9-only") as exc:
            mha_batch_prefill_func(
                torch.empty(8, 2, 64),
                torch.empty(4, 16, 2, 64),
                torch.empty(4, 16, 2, 64),
                torch.zeros(2, dtype=torch.int32),
                torch.zeros(2, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                8,
                8,
            )
        assert "gfx1100" in str(exc.value)

        q = torch.empty(8, 2, 64, dtype=torch.float16)
        k = torch.empty(4, 16, 2, 64, dtype=torch.float16)
        v = torch.empty(4, 16, 2, 64, dtype=torch.float16)
        cu = torch.zeros(2, dtype=torch.int32)
        with pytest.raises(NotImplementedError, match="gfx9-only"):
            cmdGenFunc_mha_batch_prefill(
                q,
                k,
                v,
                cu,
                cu,
                cu,
                8,
                8,
                0.0,
                0.1,
                0.0,
                False,
                False,
                -1,
                -1,
                0,
                False,
                False,
            )


def _runtime_gfx() -> str:
    try:
        from aiter.jit.utils.chip_info import get_gfx

        return get_gfx()
    except Exception:
        return ""


@pytest.mark.skipif(
    _runtime_gfx() not in ("gfx1100", "gfx1101", "gfx1102", "gfx1103"),
    reason="live factory-key check is gfx1100-1103 only",
)
def test_cmdGenFunc_fwd_emits_factory_key_gfx11():
    from aiter.ops.mha import cmdGenFunc_mha_fwd

    q, k, v = _dummy_qkv(torch.float16)
    cmds = cmdGenFunc_mha_fwd(q, k, v, 0.0, 0.1, False, -1, -1, 0, False, False)[
        "blob_gen_cmd"
    ]
    gen = cmds[0]
    assert _target_flag(gen) == "gfx11", gen
    assert "01_fmha/generate.py" in gen and "-d fwd" in gen
    assert "gfx1100" not in gen


@pytest.mark.skipif(
    _runtime_gfx() not in ("gfx1200", "gfx1201"),
    reason="live factory-key check is gfx1200-1201 only",
)
def test_cmdGenFunc_fwd_emits_factory_key_gfx12():
    from aiter.ops.mha import cmdGenFunc_mha_fwd

    q, k, v = _dummy_qkv(torch.float16)
    cmds = cmdGenFunc_mha_fwd(q, k, v, 0.0, 0.1, False, -1, -1, 0, False, False)[
        "blob_gen_cmd"
    ]
    gen = cmds[0]
    assert _target_flag(gen) == "gfx12", gen
    assert "01_fmha/generate.py" in gen and "-d fwd" in gen
    assert "gfx1201" not in gen
    assert "gfx1200" not in gen


_FMHA_MD_NAMES = {
    "module_mha_fwd",
    "module_mha_varlen_fwd",
    "module_mha_bwd",
    "module_mha_varlen_bwd",
    "module_mha_batch_prefill",
    "libmha_fwd",
    "libmha_bwd",
}


@pytest.mark.parametrize("gfx_list", [["cpu"], ["gfx1030"], []])
def test_get_args_of_build_all_skips_unmapped_fmha(gfx_list):
    """setup.py PREBUILD calls get_args_of_build('all'). Unmapped archs
    must not abort unrelated modules (REVIEW F2). This guard covers only
    the get_args_of_build("all") traversal used by PREBUILD_KERNELS 2/3;
    the PREBUILD_KERNELS==1 path in setup.py calls the mha-specific
    builders directly and still raises on an unmapped arch."""
    from aiter.jit import core as jit_core
    from aiter.jit.utils.build_targets import UnmappedCkFmhaTargetsError

    with patch("chip_info.get_gfx_list", return_value=gfx_list):
        all_ops, _ = jit_core.get_args_of_build("all")
        names = {op["md_name"] for op in all_ops}
        assert "module_activation" in names
        assert names.isdisjoint(_FMHA_MD_NAMES)
        with pytest.raises(
            UnmappedCkFmhaTargetsError, match="No CK fmha --targets mapping"
        ):
            jit_core.get_args_of_build("module_mha_fwd")
        activation = jit_core.get_args_of_build("module_activation")
        assert activation["srcs"]


def test_get_args_of_build_all_keeps_fmha_when_mapped():
    from aiter.jit import core as jit_core

    with patch("chip_info.get_gfx_list", return_value=["gfx1100"]):
        all_ops, _ = jit_core.get_args_of_build("all")
        names = {op["md_name"] for op in all_ops}
        assert "module_mha_fwd" in names
        assert "module_activation" in names
