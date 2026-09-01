"""Exact, import-free audit of the pinned MCore/TE initializer source chain.

Only explicitly supplied source checkouts are read. This module deliberately
does not import MCore, Transformer Engine, PyTorch, CuPy, or CUDA. The AST proof
is narrower than runtime parity: it proves ordinary BF16 constructor/callback/
copy ordering and seed wiring; bitwise RNG/storage parity remains a runtime gate.
"""

from __future__ import annotations

# Provenance generation is a scientific action.  It may only be imported and
# dispatched after the isolated verifier authenticates the complete package.
if __name__ == "__main__":
    raise SystemExit(
        "direct execution is forbidden; use `python -B -I "
        "/workspace/INT2__compression/INT2_Q_C/research/"
        "initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v5/verify_prelaunch.py "
        "--dispatch-source-trace ...`"
    )

import argparse
import ast
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import common


EXPECTED_FILES = {
    "mcore_pyproject": ("mcore", "pyproject.toml", "1c1837d50833f18e33fbc02d012f7aafcc99d12349a2cbba040fd4ffc7079cb5"),
    "mcore_experts": ("mcore", "megatron/core/transformer/moe/experts.py", "80f889f30cf56eefc85bc1a8908a4cb92014787f5b3a1490ee19289f3d960620"),
    "mcore_te_wrapper": ("mcore", "megatron/core/extensions/transformer_engine.py", "efc1b8e6cd9517862ec9cdd25eefb6aa2d8eceb34222b65c82bcbc3215932650"),
    "te_grouped_linear": ("te", "transformer_engine/pytorch/module/grouped_linear.py", "84d27f52ecaee38de2e324b6aa5b5fe9625129d5835183fd529f9cdeac634143"),
    "te_base": ("te", "transformer_engine/pytorch/module/base.py", "67d4a7665150761a84f8f77123f5741807e80af72469aa19bad9a9ad91704e56"),
    "mcore_initialize": ("mcore", "megatron/training/initialize.py", "76eb1beb86c18c3b96dfc97142f94b6acadd413835e207bcfbe80a36c1dbd801"),
    "mcore_rng": ("mcore", "megatron/core/tensor_parallel/random.py", "4fe12e3feab6135ec273adde452605d85434374918c6261ba05274043c16a2f0"),
}


def _regular_source_root(root: Path, label: str) -> Path:
    unresolved = common.reject_symlink_components_before_normalization(
        root, f"{label} source root", require_exists=True
    )
    info = common.lstat_or_none(unresolved)
    if info is None or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise common.ProtocolError(f"{label} source root must be a regular non-symlink directory")
    return unresolved.resolve(strict=True)


def _regular_child(root: Path, relative: str, label: str) -> Path:
    path = common.require_regular_file_before_resolve(root / relative, label)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise common.ProtocolError(f"{label} escapes supplied source root") from error
    return path


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    rows = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name]
    if len(rows) != 1:
        raise common.ProtocolError(f"expected exactly one class {name}, found {len(rows)}")
    return rows[0]


def _method(node: ast.ClassDef, name: str) -> ast.FunctionDef:
    rows = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(rows) != 1:
        raise common.ProtocolError(f"expected exactly one {node.name}.{name}, found {len(rows)}")
    return rows[0]


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    rows = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(rows) != 1:
        raise common.ProtocolError(f"expected exactly one function {name}, found {len(rows)}")
    return rows[0]


def _chain(node: ast.AST) -> str:
    names: list[str] = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
          and node.func.id == "super" and not node.args and not node.keywords):
        names.append("super")
    return ".".join(reversed(names))


def _u(node: ast.AST | None) -> str:
    return "" if node is None else ast.unparse(node)


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and _chain(n.func) == name]


def _keyword_map(call: ast.Call) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for kw in call.keywords:
        if kw.arg is not None:
            if kw.arg in result:
                raise common.ProtocolError(f"duplicate keyword {kw.arg}")
            result[kw.arg] = kw.value
    return result


def _is_fstring_i(node: ast.AST, prefix: str) -> bool:
    if not isinstance(node, ast.JoinedStr) or len(node.values) != 2:
        return False
    a, b = node.values
    return (isinstance(a, ast.Constant) and a.value == prefix
            and isinstance(b, ast.FormattedValue) and isinstance(b.value, ast.Name)
            and b.value.id == "i")


def _range_self_num_gemms(loop: ast.For) -> bool:
    return (isinstance(loop.target, ast.Name) and loop.target.id == "i"
            and isinstance(loop.iter, ast.Call) and _chain(loop.iter.func) == "range"
            and len(loop.iter.args) == 1 and _u(loop.iter.args[0]) == "self.num_gemms")


def _one_call(node: ast.AST, name: str, label: str) -> ast.Call:
    rows = _calls(node, name)
    if len(rows) != 1:
        raise common.ProtocolError(f"expected one {label}, found {len(rows)}")
    return rows[0]


def _assignment_calls(function: ast.FunctionDef, targets: set[str]) -> dict[str, ast.Call]:
    result: dict[str, ast.Call] = {}
    for row in ast.walk(function):
        if not isinstance(row, ast.Assign) or len(row.targets) != 1 or not isinstance(row.value, ast.Call):
            continue
        target = _chain(row.targets[0])
        if target in targets:
            if target in result:
                raise common.ProtocolError(f"duplicate assignment call for {target}")
            result[target] = row.value
    if set(result) != targets:
        raise common.ProtocolError(f"missing assignment calls: {sorted(targets-set(result))}")
    return result


def _audit_mcore_pyproject(text: str) -> dict[str, Any]:
    versions = re.findall(r'(?ms)^\[\[tool\.uv\.dependency-metadata\]\]\s*\nname\s*=\s*"transformer-engine"\s*\nversion\s*=\s*"([^"]+)"', text)
    revisions = re.findall(r'(?m)^transformer-engine\s*=\s*\{\s*git\s*=\s*"https://github\.com/NVIDIA/TransformerEngine\.git",\s*rev\s*=\s*"([0-9a-f]{40})"\s*\}\s*$', text)
    if versions != [common.TE_SOURCE_VERSION] or revisions != [common.TE_REVISION]:
        raise common.ProtocolError("MCore pyproject TE version/revision binding changed")
    return {"dependency_metadata_version": versions[0], "git_revision": revisions[0],
            "pypi_2_18_0_allowed_only_after_all_seven_source_hashes_match": True}


def _audit_mcore_experts(text: str) -> dict[str, Any]:
    method = _method(_class(ast.parse(text), "TEGroupedMLP"), "__init__")
    super_call = _one_call(method, "super.__init__", "TEGroupedMLP super constructor")
    if {k: _u(v) for k, v in _keyword_map(super_call).items()} != {"config": "config"}:
        raise common.ProtocolError("TEGroupedMLP super(config=config) binding changed")
    calls = _assignment_calls(method, {"self.linear_fc1", "self.linear_fc2"})
    fc1, fc2 = calls["self.linear_fc1"], calls["self.linear_fc2"]
    if not super_call.lineno < fc1.lineno < fc2.lineno:
        raise common.ProtocolError("TEGroupedMLP super/FC1/FC2 construction order changed")
    if _chain(fc1.func) != "submodules.linear_fc1" or _chain(fc2.func) != "submodules.linear_fc2":
        raise common.ProtocolError("TEGroupedMLP builder binding changed")
    if len(fc1.args) != 3 or len(fc2.args) != 3:
        raise common.ProtocolError("TEGroupedMLP grouped-linear positional arity changed")
    if _u(fc1.args[0]) != "self.num_local_experts" or _u(fc2.args[0]) != "self.num_local_experts":
        raise common.ProtocolError("TEGroupedMLP local-expert count binding changed")
    if _u(fc1.args[2]) != "ffn_hidden_size" or _u(fc2.args[1]) != "not_none(self.config.moe_ffn_hidden_size)":
        raise common.ProtocolError("TEGroupedMLP projection width binding changed")
    gated = [n for n in ast.walk(method) if isinstance(n, ast.AugAssign)
             and isinstance(n.op, ast.Mult) and _u(n.target) == "ffn_hidden_size"
             and isinstance(n.value, ast.Constant) and n.value.value == 2 and n.lineno < fc1.lineno]
    if len(gated) != 1:
        raise common.ProtocolError("exact pre-FC1 gated width doubling was not found")
    k1, k2 = _keyword_map(fc1), _keyword_map(fc2)
    for key, value in {"config": "self.config", "bias": "self.config.add_bias_linear", "is_expert": "True", "pg_collection": "pg_collection"}.items():
        if _u(k1.get(key)) != value or _u(k2.get(key)) != value:
            raise common.ProtocolError(f"TEGroupedMLP {key} wrapper binding changed")
    if _u(k1.get("init_method")) != "not_none(self.config.init_method)" or _u(k2.get("init_method")) != "not_none(self.config.output_layer_init_method)":
        raise common.ProtocolError("FC1/FC2 initializer callback binding changed")
    return {"class": "TEGroupedMLP", "super_line": super_call.lineno,
            "fc1_line": fc1.lineno, "fc2_line": fc2.lineno,
            "gated_width_line": gated[0].lineno,
            "proved_order": ["MegatronModule.__init__", "linear_fc1", "linear_fc2"]}


def _audit_mcore_wrapper(text: str) -> dict[str, Any]:
    method = _method(_class(ast.parse(text), "TEGroupedLinear"), "__init__")
    super_call = _one_call(method, "super.__init__", "TEGroupedLinear TE super constructor")
    kw = _keyword_map(super_call)
    expected = {"num_gemms": "num_gemms", "in_features": "input_size", "out_features": "output_size",
                "tp_size": "tp_size", "init_method": "condition_init_method(config, init_method)",
                "bias": "bias", "return_bias": "self.te_return_bias", "parallel_mode": "parallel_mode"}
    for key, value in expected.items():
        if _u(kw.get(key)) != value:
            raise common.ProtocolError(f"TEGroupedLinear super keyword {key} changed")
    tracker = [n for n in ast.walk(method) if isinstance(n, ast.Assign) and len(n.targets) == 1
               and _u(n.targets[0]) == "extra_kwargs['rng_tracker_name']"
               and isinstance(n.value, ast.Call) and _chain(n.value.func) == "get_expert_parallel_rng_tracker_name"]
    if len(tracker) != 1 or tracker[0].lineno >= super_call.lineno:
        raise common.ProtocolError("expert RNG tracker wrapper edge changed")
    grouped: dict[str, ast.Assign] = {}
    for n in ast.walk(method):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and _u(n.targets[0]) in {"extra_kwargs['single_grouped_weight']", "extra_kwargs['single_grouped_bias']"}:
            grouped[_u(n.targets[0])] = n
    if len(grouped) != 2:
        raise common.ProtocolError("single-grouped wrapper configuration edges missing")
    for suffix in ("weight", "bias"):
        node = grouped[f"extra_kwargs['single_grouped_{suffix}']"]
        if node.lineno >= super_call.lineno or _u(node.value) != f"getattr(config, 'moe_single_grouped_{suffix}', False)":
            raise common.ProtocolError(f"single_grouped_{suffix} wrapper binding changed")
    if "get_cuda_rng_tracker" not in _u(kw.get("get_rng_state_tracker")):
        raise common.ProtocolError("TE wrapper did not pass the CUDA RNG tracker")
    return {"class": "TEGroupedLinear", "expert_tracker_assignment_line": tracker[0].lineno,
            "te_super_line": super_call.lineno, "passes_num_gemms_and_exact_widths": True,
            "passes_conditioned_init_callback": True, "passes_expert_rng_tracker": True,
            "passes_single_grouped_flags_via_extra_kwargs": True}


def _registration_loop(init: ast.FunctionDef) -> tuple[ast.For, ast.Call]:
    hits: list[tuple[ast.For, ast.Call]] = []
    for loop in [n for n in ast.walk(init) if isinstance(n, ast.For) and _range_self_num_gemms(n)]:
        for call in _calls(loop, "self.register_parameter"):
            if call.args and _is_fstring_i(call.args[0], "weight") and _u(_keyword_map(call).get("init_fn")) == "init_method":
                hits.append((loop, call))
    if len(hits) != 1:
        raise common.ProtocolError(f"expected one structural numbered-weight registration loop, found {len(hits)}")
    return hits[0]


def _audit_grouped_linear(text: str) -> dict[str, Any]:
    cls = _class(ast.parse(text), "GroupedLinear")
    init = _method(cls, "__init__")
    args = {n.arg for n in init.args.args + init.args.kwonlyargs}
    if "parameters_split" in args or "single_grouped_weight" not in args:
        raise common.ProtocolError("GroupedLinear constructor ABI changed")
    loop, registration = _registration_loop(init)
    rendered = _u(registration)
    for token in ("self.out_features", "self.in_features", "get_rng_state_tracker=get_rng_state_tracker"):
        if token not in rendered:
            raise common.ProtocolError(f"numbered-weight registration lost {token}")
    reset_calls = _calls(init, "self.reset_parameters")
    if len(reset_calls) != 1:
        raise common.ProtocolError("GroupedLinear constructor reset call is not unique")
    constructor_reset = reset_calls[0]
    if _u(_keyword_map(constructor_reset).get("defer_init")) != "is_meta" or constructor_reset.lineno <= (loop.end_lineno or loop.lineno):
        raise common.ProtocolError("constructor reset no longer follows all numbered registration")
    all_num_loops = [n for n in ast.walk(init) if isinstance(n, ast.For) and _range_self_num_gemms(n)]
    if len(all_num_loops) < 2:
        raise common.ProtocolError("pinned two-loop constructor regression sentinel missing")
    reset = _method(cls, "reset_parameters")
    base_reset = _one_call(reset, "super.reset_parameters", "base reset call")
    pack_call = _one_call(reset, "self.make_grouped_weights", "copy-pack call")
    if not base_reset.lineno < pack_call.lineno:
        raise common.ProtocolError("copy-pack no longer follows base numbered reset")
    if _u(_keyword_map(base_reset).get("defer_init")) != "defer_init" or _u(_keyword_map(pack_call).get("defer_init")) != "defer_init":
        raise common.ProtocolError("reset/copy-pack defer_init binding changed")
    if not any(isinstance(n, ast.If) and _u(n.test) == "self.single_grouped_weight" and pack_call in list(ast.walk(n)) for n in ast.walk(reset)):
        raise common.ProtocolError("make_grouped_weights is not gated by single_grouped_weight")
    make = _method(cls, "make_grouped_weights")
    weights_rows = [n for n in ast.walk(make) if isinstance(n, ast.Assign) and len(n.targets) == 1
                    and _u(n.targets[0]) == "weights" and "getattr(self, f'weight{i}')" in _u(n.value)
                    and "range(self.num_gemms)" in _u(n.value)]
    if len(weights_rows) != 1:
        raise common.ProtocolError("numbered-weight collection edge changed")
    ordinary_copies = [n for n in ast.walk(make) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr == "copy_"
                       and _u(n.func.value) == "grouped_weights.quantized_tensors[i]"
                       and len(n.args) == 1 and _u(n.args[0]) == "weights[i]"]
    if len(ordinary_copies) != 1:
        raise common.ProtocolError("ordinary BF16 numbered-to-grouped copy edge changed")
    grouped_register = [n for n in _calls(make, "self.register_parameter")
                        if n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "weight"]
    if len(grouped_register) != 1 or _u(_keyword_map(grouped_register[0]).get("init_fn")) != "self.init_method":
        raise common.ProtocolError("single grouped weight registration edge changed")
    clear_loops: list[ast.For] = []
    for candidate in [n for n in ast.walk(make) if isinstance(n, ast.For) and _range_self_num_gemms(n)]:
        clears = [c for c in _calls(candidate, "self.register_parameter") if len(c.args) >= 2
                  and _is_fstring_i(c.args[0], "weight") and isinstance(c.args[1], ast.Constant)
                  and c.args[1].value is None]
        if clears:
            clear_loops.append(candidate)
    if len(clear_loops) != 1:
        raise common.ProtocolError("numbered-weight clearing loop changed")
    if not weights_rows[0].lineno < ordinary_copies[0].lineno < grouped_register[0].lineno < clear_loops[0].lineno:
        raise common.ProtocolError("collect/copy/register/clear copy-pack order changed")
    return {"class": "GroupedLinear", "constructor_registration_line": loop.lineno,
            "constructor_range_num_gemms_loop_count": len(all_num_loops),
            "constructor_reset_line": constructor_reset.lineno,
            "base_reset_line": base_reset.lineno, "copy_pack_dispatch_line": pack_call.lineno,
            "ordinary_copy_line": ordinary_copies[0].lineno,
            "single_register_line": grouped_register[0].lineno,
            "numbered_clear_line": clear_loops[0].lineno,
            "proved_ordinary_bf16_order": ["register_numbered", "constructor_reset", "base_init_callbacks", "copy_numbered_to_grouped", "register_grouped", "clear_numbered"]}


def _audit_base(text: str) -> dict[str, Any]:
    tree = ast.parse(text)
    hits: list[tuple[ast.ClassDef, ast.FunctionDef, ast.For]] = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for method in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "reset_parameters"]:
            for loop in [n for n in ast.walk(method) if isinstance(n, ast.For)
                         and isinstance(n.iter, ast.Call) and _chain(n.iter.func) == "self.named_parameters"
                         and _u(_keyword_map(n.iter).get("recurse")) == "False"]:
                if _calls(loop, "init_fn"):
                    hits.append((cls, method, loop))
    if len(hits) != 1:
        raise common.ProtocolError("TE base nonrecursive initialization loop is ambiguous")
    cls, method, loop = hits[0]
    callbacks = [c for c in _calls(loop, "init_fn") if len(c.args) == 1 and _u(c.args[0]) == "param"]
    if len(callbacks) != 3:
        raise common.ProtocolError("TE base must contain one direct and two tracker-fork init callbacks")
    meta = [n for n in ast.walk(loop) if isinstance(n, ast.Assign) and len(n.targets) == 1
            and _u(n.targets[0]) == "init_fn" and _u(n.value) == "self.param_init_meta[name].init_fn"]
    if len(meta) != 1 or meta[0].lineno >= min(c.lineno for c in callbacks):
        raise common.ProtocolError("TE base init_fn lookup/callback order changed")
    return {"class": cls.name, "reset_parameters_line": method.lineno,
            "named_parameters_loop_line": loop.lineno,
            "direct_and_tracker_callback_lines": sorted(c.lineno for c in callbacks),
            "nonrecursive_registration_order_iteration": True}


def _audit_initialize(text: str) -> dict[str, Any]:
    fn = _function(ast.parse(text), "_set_random_seed")
    guards = [n for n in fn.body if isinstance(n, ast.If)
              and "seed_ is not None" in _u(n.test) and "seed_ > 0" in _u(n.test)]
    if len(guards) != 1:
        raise common.ProtocolError("positive CLI seed guard changed")
    guard = guards[0]
    else_tree = ast.Module(body=guard.orelse, type_ignores=[])
    raises = [n for n in ast.walk(else_tree) if isinstance(n, ast.Raise)]
    if len(raises) != 1 or "positive integer" not in _u(raises[0]):
        raise common.ProtocolError("nonpositive seed rejection changed")
    body = ast.Module(body=guard.body, type_ignores=[])
    pp_assign = [n for n in ast.walk(body) if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and _u(n.targets[0]) == "pp_rank" and "get_pg_rank(pp_group)" in _u(n.value)
                 and "mpu.get_pipeline_model_parallel_rank()" in _u(n.value)]
    seed_assign = [n for n in ast.walk(body) if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and _u(n.targets[0]) == "seed" and _u(n.value) == "seed_ + 100 * pp_rank"]
    calls = _calls(body, "tensor_parallel.model_parallel_cuda_manual_seed")
    if len(pp_assign) != 1 or len(seed_assign) != 1 or len(calls) != 1:
        raise common.ProtocolError("pipeline seed-to-CUDA call chain changed")
    call, kw = calls[0], _keyword_map(calls[0])
    if not call.args or _u(call.args[0]) != "seed":
        raise common.ProtocolError("manual CUDA seed first argument changed")
    for key in ("tp_rank", "ep_rank", "etp_rank"):
        if _u(kw.get(key)) != key:
            raise common.ProtocolError(f"manual CUDA seed {key} binding changed")
    if not guard.lineno < pp_assign[0].lineno < seed_assign[0].lineno < call.lineno:
        raise common.ProtocolError("positive-guard/pipeline-adjust/CUDA-seed ordering changed")
    return {"function": "_set_random_seed", "positive_guard_line": guard.lineno,
            "pp_rank_line": pp_assign[0].lineno, "pipeline_seed_line": seed_assign[0].lineno,
            "manual_cuda_seed_line": call.lineno, "nonpositive_seed_raises": True,
            "stored_u16_to_cli_rule": "cli_seed = stored_seed_u16 + 1"}


def _audit_rng(text: str) -> dict[str, Any]:
    fn = _function(ast.parse(text), "model_parallel_cuda_manual_seed")
    assignments = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and _u(n.targets[0]) == "expert_parallel_seed"]
    if len(assignments) != 1 or _u(assignments[0].value) != "seed + 1024 + 100 * ep_rank + etp_rank":
        raise common.ProtocolError("expert-parallel seed formula changed")
    adds = [c for c in _calls(fn, "_CUDA_RNG_STATE_TRACKER.add") if len(c.args) == 2
            and _u(c.args[0]) == "_EXPERT_PARALLEL_RNG_TRACKER_NAME"
            and _u(c.args[1]) == "expert_parallel_seed"]
    if len(adds) != 1 or adds[0].lineno <= assignments[0].lineno:
        raise common.ProtocolError("expert RNG tracker registration changed")
    fallback: dict[str, ast.Assign] = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and _u(n.targets[0]) in {"ep_rank", "etp_rank"}:
            fallback[_u(n.targets[0])] = n
    for name, phrase in (("ep_rank", "get_expert_model_parallel_rank"), ("etp_rank", "get_expert_tensor_parallel_rank")):
        if name not in fallback or phrase not in _u(fallback[name].value):
            raise common.ProtocolError(f"{name} fallback binding changed")
    return {"function": "model_parallel_cuda_manual_seed", "expert_seed_line": assignments[0].lineno,
            "tracker_add_line": adds[0].lineno,
            "expert_seed_formula": "pipeline_seed + 1024 + 100*ep_rank + etp_rank",
            "explicit_or_mpu_ep_etp_rank_binding": True}


def _logical_trace_digest() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ep in common.EP_SIZES:
        local_experts = 128 // ep
        for etp in common.ETP_SIZES:
            h = common.ROWS // etp
            events = ([{"projection": "fc1", "local_expert": e, "shape": [2 * h, common.COLUMNS]} for e in range(local_experts)]
                      + [{"projection": "fc2", "local_expert": e, "shape": [common.COLUMNS, h]} for e in range(local_experts)])
            rows.append({"ep": ep, "etp": etp, "local_experts": local_experts,
                         "call_count": len(events),
                         "events_sha256": common.sha256_bytes(common.canonical_json_bytes(events)),
                         "first": events[0], "fc2_first": events[local_experts], "last": events[-1]})
    return {"kind": "procedural_geometry_emulation_not_full_pre_layer_lifecycle_proof",
            "geometries": rows, "geometry_count": len(rows),
            "table_sha256": common.sha256_bytes(common.canonical_json_bytes(rows))}


def audit_source_texts(texts: Mapping[str, str]) -> dict[str, Any]:
    if set(texts) != set(EXPECTED_FILES):
        raise common.ProtocolError("source-text label set does not equal the seven-file pin")
    return {"mcore_dependency_pin": _audit_mcore_pyproject(texts["mcore_pyproject"]),
            "mcore_grouped_mlp": _audit_mcore_experts(texts["mcore_experts"]),
            "mcore_te_wrapper": _audit_mcore_wrapper(texts["mcore_te_wrapper"]),
            "te_grouped_linear": _audit_grouped_linear(texts["te_grouped_linear"]),
            "te_base_reset": _audit_base(texts["te_base"]),
            "mcore_seed_setup": _audit_initialize(texts["mcore_initialize"]),
            "mcore_expert_rng": _audit_rng(texts["mcore_rng"])}


def audit_sources(mcore_root: Path, te_root: Path) -> dict[str, Any]:
    if common.environment_has_cuda_imports():
        raise common.ProtocolError("source trace must run before forbidden runtime imports")
    roots = {
        "mcore": _regular_source_root(mcore_root, "MCore"),
        "te": _regular_source_root(te_root, "Transformer Engine"),
    }
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    texts: dict[str, str] = {}
    for label, (root_key, relative, expected) in EXPECTED_FILES.items():
        path = _regular_child(roots[root_key], relative, label)
        observed = common.sha256_file(path)
        if observed != expected:
            raise common.ProtocolError(f"{label} SHA-256 mismatch")
        paths[label], hashes[label], texts[label] = path, observed, path.read_text(encoding="utf-8")
    semantic = audit_source_texts(texts)
    result: dict[str, Any] = {
        "schema": "qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_source_trace_v2",
        "status": "PASS_EXACT_SEVEN_FILE_SOURCE_TRACE_RUNTIME_PARITY_STILL_REQUIRED",
        "mcore_revision": common.MCORE_REVISION,
        "transformer_engine_revision": common.TE_REVISION,
        "transformer_engine_source_version": common.TE_SOURCE_VERSION,
        "transformer_engine_pypi_version_policy": {"version": common.TE_PYPI_VERSION, "accepted_only_if_all_runtime_source_files_rehash_to_this_receipt": True},
        "files": {label: {"relative_path": EXPECTED_FILES[label][1], "sha256": hashes[label]} for label in sorted(paths)},
        "semantic_edges": semantic,
        "procedural_geometry_trace": _logical_trace_digest(),
        "claim_boundary": {
            "source_proves_ordinary_bf16_fc1_all_then_fc2_all_constructor_callback_order": True,
            "source_proves_numbered_then_copy_pack_order": True,
            "copy_pack_bitwise_and_terminal_rng_parity_source_only": False,
            "pytorch_cupy_philox_parity_source_only": False,
            "direct_bf16_vs_cast_parity_source_only": False,
            "numeric_full_pre_layer_15_expert_rng_lifecycle_source_only": False,
            "pp_cross_seed_equivalence_used": False,
        },
        "access_attestation": {"qwen_payload_or_manifest_opened_statted_or_hashed": False,
                               "cuda_or_forbidden_runtime_imported": False,
                               "only_the_seven_explicit_source_files_opened": True},
        "execution_boundary": None,
    }
    result["receipt_sha256"] = common.sha256_bytes(common.canonical_json_bytes(result))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcore-root", type=Path, required=True)
    parser.add_argument("--te-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    guard = common.BoundaryGuard(
        "SOURCE_TRACE_CREATE_ONCE",
        outputs=(("source-trace output", args.output, "file", False),),
        inputs=(
            ("MCore source root", args.mcore_root, "directory"),
            ("Transformer Engine source root", args.te_root, "directory"),
        ),
    )
    result = audit_sources(args.mcore_root, args.te_root)
    normalized = dict(result)
    normalized.pop("receipt_sha256")
    normalized["execution_boundary"] = guard.receipt()
    normalized["receipt_sha256"] = common.sha256_bytes(
        common.canonical_json_bytes(normalized)
    )
    result = normalized
    guard.revalidate("immediately before source-trace create-new")
    output = common.write_json_create_new(
        args.output, result, "source-trace output"
    )
    print(output)
    return 0
