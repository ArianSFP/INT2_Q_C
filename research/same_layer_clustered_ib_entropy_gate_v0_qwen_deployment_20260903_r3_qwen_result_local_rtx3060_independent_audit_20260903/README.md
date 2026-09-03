# Independent audit: local RTX 3060 CBIB-r3 Qwen result

Verdict: `PASS_COMPLETED_CHILD_RESULT_WITH_HARMLESS_STDERR_WARNING__HARD_KILL_CBIB_FIXED_LABEL`.

This package independently authenticates the six preserved execution files from the consumed local CBIB-r3 authority and binds them to the byte-exact frozen deployment and local capability closures. It does not read Qwen payloads, use CUDA/CuPy, access a network, or re-execute the one-use payload path.

The child completed normally enough to serialize `result.json`, emit its exact SHA-256 in its sole terminal stdout receipt, and reach the terminal scientific status `HARD_KILL_CHARGED_OR_CONTROLS_BELOW_TARGET`. The wrapper subsequently rejected the run because it required literally empty stderr. The exact stderr is only CuPy's `CUDA path could not be detected` `UserWarning`; the authenticated computation nevertheless used CuPy 14.2.0 on the pinned RTX 3060 and recorded CUDA runtime/driver APIs 12090/12060. This audit does not relabel the wrapper as a wrapper success: its failure and consumed authority remain literal.

Scientifically, group-2 common/private coding showed a favourable-looking 0.4597951000 bpw private-stream saving, but the necessary common stream cancelled it: net ideal gain was only 0.00001073076 bpw, and the fully charged gain was -0.0035780649 bpw. All four group sizes had negative charged gain. The ideal page-layout envelopes were below 2x at both rate endpoints, so read amplification was not the reason for rejection. This hard-kills this fixed-label CBIB formulation on the frozen Qwen layer-15 Up/Down.T panel; it does not close flexible-label joint quantization or other same-layer conditional models.

Run the evidence verifier with a stdlib Python:

```text
python -I -B verify_evidence.py --audit-package . --deployment-package ../same_layer_clustered_ib_entropy_gate_v0_qwen_deployment_20260903_r3 --capability-package ../same_layer_clustered_ib_entropy_gate_v0_qwen_deployment_20260903_r3_local_rtx3060_capability_20260903_r3
```

Then verify the sealed package with the published manifest digest:

```text
python -I -B verify_audit.py --audit-package . --audit-manifest-sha256 <PUBLISHED_SHA256>
```
