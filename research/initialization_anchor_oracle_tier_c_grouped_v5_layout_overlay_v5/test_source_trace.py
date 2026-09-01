from __future__ import annotations

import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import common
import source_trace


PYPROJECT = '''
[[tool.uv.dependency-metadata]]
name = "transformer-engine"
version = "2.18.0+27486e03"
requires-dist = []
[tool.uv.sources]
transformer-engine = { git = "https://github.com/NVIDIA/TransformerEngine.git", rev = "27486e03cfc1fa41f6932dcecdc47c71c47eac3e" }
'''

EXPERTS = '''
class TEGroupedMLP:
    def __init__(self, num_local_experts, config, submodules, pg_collection=None, name=None):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.input_size = self.config.hidden_size
        ffn_hidden_size = not_none(self.config.moe_ffn_hidden_size)
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2
        self.linear_fc1 = submodules.linear_fc1(
            self.num_local_experts, self.input_size, ffn_hidden_size,
            config=self.config, init_method=not_none(self.config.init_method),
            bias=self.config.add_bias_linear, skip_bias_add=False, is_expert=True,
            tp_comm_buffer_name='fc1', pg_collection=pg_collection)
        self.linear_fc2 = submodules.linear_fc2(
            self.num_local_experts, not_none(self.config.moe_ffn_hidden_size), self.config.hidden_size,
            config=self.config, init_method=not_none(self.config.output_layer_init_method),
            bias=self.config.add_bias_linear, skip_bias_add=True, is_expert=True,
            tp_comm_buffer_name='fc2', pg_collection=pg_collection)
'''

WRAPPER = '''
class TEGroupedLinear:
    def __init__(self, num_gemms, input_size, output_size, *, parallel_mode, config,
                 init_method, bias, skip_bias_add, is_expert=False, pg_collection=None):
        extra_kwargs = {}
        self.te_return_bias = skip_bias_add and bias
        if is_expert:
            extra_kwargs["rng_tracker_name"] = get_expert_parallel_rng_tracker_name()
        if is_te_min_version("2.14.0"):
            if "single_grouped_weight" in grouped_linear_init_params:
                extra_kwargs["single_grouped_weight"] = getattr(config, "moe_single_grouped_weight", False)
            if "single_grouped_bias" in grouped_linear_init_params:
                extra_kwargs["single_grouped_bias"] = getattr(config, "moe_single_grouped_bias", False)
        tp_size = 1
        with context():
            super().__init__(
                num_gemms=num_gemms, in_features=input_size, out_features=output_size,
                sequence_parallel=config.sequence_parallel, tp_group=None, tp_size=tp_size,
                get_rng_state_tracker=(get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None),
                init_method=condition_init_method(config, init_method), bias=bias,
                return_bias=self.te_return_bias, parallel_mode=parallel_mode, **extra_kwargs)
'''

GROUPED = '''
class GroupedLinear:
    def __init__(self, num_gemms, in_features, out_features, init_method=None,
                 get_rng_state_tracker=None, single_grouped_weight=False, device="cuda"):
        self.num_gemms = num_gemms
        self.out_features = out_features
        self.in_features = in_features
        self.single_grouped_weight = single_grouped_weight
        for i in range(self.num_gemms):
            self.register_parameter(
                f"weight{i}", torch.nn.Parameter(torch.empty(self.out_features, self.in_features)),
                init_fn=init_method, get_rng_state_tracker=get_rng_state_tracker)
        is_meta = torch.device(device).type == "meta"
        self.reset_parameters(defer_init=is_meta)
        for i in range(self.num_gemms):
            if name in (f"weight{i}", f"bias{i}"):
                touch(name)

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)
        if self.single_grouped_weight:
            self.make_grouped_weights(defer_init=defer_init)

    def make_grouped_weights(self, defer_init=False):
        if defer_init:
            return
        weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
        grouped_weights = make_storage(weights)
        for i in range(self.num_gemms):
            if primary:
                grouped_weights.quantized_tensors[i].copy_from_storage(weights[i])
            else:
                grouped_weights.quantized_tensors[i].copy_(weights[i])
        self.register_parameter("weight", Parameter(grouped_weights), init_fn=self.init_method)
        for i in range(self.num_gemms):
            self.register_parameter(f"weight{i}", None)
'''

BASE = '''
class TransformerEngineBaseModule:
    def reset_parameters(self, defer_init=False):
        if defer_init:
            return
        for name, param in self.named_parameters(recurse=False):
            init_fn = self.param_init_meta[name].init_fn
            tracker = self.param_init_meta[name].get_rng_state_tracker
            if tracker is None:
                init_fn(param)
            else:
                if self.rng_tracker_name:
                    with tracker().fork(self.rng_tracker_name):
                        init_fn(param)
                else:
                    with tracker().fork():
                        init_fn(param)
'''

INITIALIZE = '''
def _set_random_seed(seed_, pp_group=None, tp_group=None, ep_group=None, etp_group=None):
    if seed_ is not None and seed_ > 0:
        pp_rank = get_pg_rank(pp_group) if pp_group is not None else mpu.get_pipeline_model_parallel_rank()
        seed = seed_ + (100 * pp_rank)
        tp_rank = ep_rank = etp_rank = None
        tensor_parallel.model_parallel_cuda_manual_seed(
            seed, False, False, False, tp_rank=tp_rank, ep_rank=ep_rank, etp_rank=etp_rank)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed_))
'''

RNG = '''
def model_parallel_cuda_manual_seed(seed, ep_rank=None, etp_rank=None):
    if ep_rank is None:
        ep_rank = get_expert_model_parallel_rank()
    if etp_rank is None:
        etp_rank = get_expert_tensor_parallel_rank()
    expert_parallel_seed = seed + 1024 + 100 * ep_rank + etp_rank
    _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)
'''


def fixture():
    return {"mcore_pyproject": PYPROJECT, "mcore_experts": EXPERTS,
            "mcore_te_wrapper": WRAPPER, "te_grouped_linear": GROUPED,
            "te_base": BASE, "mcore_initialize": INITIALIZE, "mcore_rng": RNG}


class SourceTraceV5LayoutOverlayTests(unittest.TestCase):
    def test_direct_source_trace_execution_fails_before_package_imports(self):
        package = Path(source_trace.__file__).resolve().parent
        result = subprocess.run(
            [sys.executable, "-B", "-I", str(package / "source_trace.py")],
            cwd=package,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("direct execution is forbidden", result.stderr)
        self.assertFalse((package / "__pycache__").exists())

    def test_complete_seven_file_trace_and_real_two_loop_regression(self):
        result = source_trace.audit_source_texts(fixture())
        grouped = result["te_grouped_linear"]
        self.assertEqual(grouped["constructor_range_num_gemms_loop_count"], 2)
        self.assertLess(grouped["constructor_registration_line"], grouped["constructor_reset_line"])
        self.assertTrue(result["mcore_te_wrapper"]["passes_expert_rng_tracker"])
        self.assertTrue(result["mcore_seed_setup"]["nonpositive_seed_raises"])

    def assert_mutation_rejected(self, label, old, new):
        texts = fixture()
        self.assertIn(old, texts[label])
        texts[label] = texts[label].replace(old, new, 1)
        with self.assertRaises(common.ProtocolError):
            source_trace.audit_source_texts(texts)

    def test_missing_registration_init_edge_rejected(self):
        self.assert_mutation_rejected("te_grouped_linear", "init_fn=init_method", "init_fn=other")

    def test_missing_constructor_reset_edge_rejected(self):
        self.assert_mutation_rejected("te_grouped_linear", "self.reset_parameters(defer_init=is_meta)", "touch(is_meta)")

    def test_reversed_base_reset_copy_pack_rejected(self):
        old = "super().reset_parameters(defer_init=defer_init)\n        if self.single_grouped_weight:\n            self.make_grouped_weights(defer_init=defer_init)"
        new = "self.make_grouped_weights(defer_init=defer_init)\n        if self.single_grouped_weight:\n            super().reset_parameters(defer_init=defer_init)"
        self.assert_mutation_rejected("te_grouped_linear", old, new)

    def test_copy_register_clear_edge_rejected(self):
        self.assert_mutation_rejected("te_grouped_linear", ".copy_(weights[i])", ".copy_(weights[0])")

    def test_mcore_fc_order_edge_rejected(self):
        self.assert_mutation_rejected("mcore_experts", "self.linear_fc1 =", "self.linear_other =")

    def test_wrapper_num_gemms_edge_rejected(self):
        self.assert_mutation_rejected("mcore_te_wrapper", "num_gemms=num_gemms", "num_gemms=1")

    def test_wrapper_expert_tracker_edge_rejected(self):
        self.assert_mutation_rejected("mcore_te_wrapper", "get_expert_parallel_rng_tracker_name()", "get_cuda_rng_tracker()")

    def test_positive_seed_guard_edge_rejected(self):
        self.assert_mutation_rejected("mcore_initialize", "seed_ > 0", "seed_ >= 0")

    def test_pipeline_shift_edge_rejected(self):
        self.assert_mutation_rejected("mcore_initialize", "100 * pp_rank", "99 * pp_rank")

    def test_expert_seed_formula_edge_rejected(self):
        self.assert_mutation_rejected("mcore_rng", "100 * ep_rank", "99 * ep_rank")

    def test_te_pin_edge_rejected(self):
        self.assert_mutation_rejected("mcore_pyproject", "2.18.0+27486e03", "2.18.0")

    def test_missing_seventh_file_rejected(self):
        texts = fixture()
        del texts["mcore_rng"]
        with self.assertRaises(common.ProtocolError):
            source_trace.audit_source_texts(texts)

    def test_source_trace_output_dangling_symlink_rejected_before_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "must-not-be-created.json"
            output = root / "dangling-output.json"
            try:
                output.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            with mock.patch.object(
                source_trace, "audit_sources", side_effect=AssertionError("audit reached")
            ):
                with self.assertRaises(common.ProtocolError):
                    source_trace.main([
                        "--mcore-root", str(root / "missing-mcore"),
                        "--te-root", str(root / "missing-te"),
                        "--output", str(output),
                    ])
            self.assertFalse(target.exists())

    def test_source_root_and_leaf_symlinks_rejected_before_resolve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_root = root / "real"
            real_root.mkdir()
            source = real_root / "source.py"
            source.write_text("x = 1\n", encoding="utf-8")
            root_link = root / "root-link"
            leaf_link = real_root / "leaf-link.py"
            try:
                root_link.symlink_to(real_root, target_is_directory=True)
                leaf_link.symlink_to(source)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            with mock.patch.object(Path, "resolve", side_effect=AssertionError("resolve reached")):
                with self.assertRaises(common.ProtocolError):
                    source_trace._regular_source_root(root_link, "fixture")
                with self.assertRaises(common.ProtocolError):
                    source_trace._regular_child(real_root, "leaf-link.py", "fixture")


if __name__ == "__main__":
    unittest.main()
