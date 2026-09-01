# Independent source audit: free-order SwiGLU path oracle v1

Verdict: **BLOCK**

This is a source-only scientific reproducibility audit of the frozen producer directory `free_order_swiglu_path_oracle_v1`. The producer was not modified. No Qwen/model tensor, pinned-panel datum, validation datum, CuPy module, CUDA API, or GPU was accessed.

The package's own source-only checks pass: all 11 unit tests pass on Linux/Python 3.12.3 and `verify_package.py` reports PASS with 128 checks. Its exact seven-file closure and all producer hashes also reproduce byte-for-byte on Windows and Linux.

Those mechanical passes do not authorize payload execution. The stage-0 early-kill premise is false for the stated stage-1 family. Stage 0 diagonalizes three marginal 768-by-768 neuron covariances separately, one for each role. Stage 1 permits a full 3-by-3 predecessor-to-target regression, including off-diagonal Gate/Up/Down terms. Cross-neuron, cross-role covariance is absent from all three stage-0 marginal spectra, so the later family is not contained by the earlier envelope.

An exact counterexample fits the frozen geometry. Choose 768 orthonormal, zero-mean vectors `e_i` in the 2047-dimensional zero-mean subspace of the 2048 model coordinates. For neuron `i`, set its three role rows to `(e_i, e_(i+1), e_(i+2))`, with indices modulo 768. Every separate-role neuron covariance has a flat spectrum, so the separate-role KLT has `F=1` and `s=0` at every tested rate. Along the legal path `0,1,...,767`, however, the full 3-by-3 predictor captures two energy units on each of 767 edges. Its residual ratio is `770/2304 = 0.3342013888888889`, giving `s=0.7906051829300244 bpw`, well above both the required `0.16096404744368115 bpw` and the strongest frozen full-3-by-3 FP16 side-adjusted requirement `0.1858070514584381 bpw`.

Therefore a stage-0 failure cannot safely kill stage 1, and this audit blocks every Qwen/model payload execution under v1.

The minimal distinct-successor repair is to refreeze a v2 whose stage-0 favorable envelope is one free joint 2304-axis role-neuron KLT (not three marginal 768-axis KLTs), with the same matched controls and statistical gates. A simpler valid routing repair is to remove the stage-0 early kill and always evaluate the already specified full cross-role stage 1 after separate authorization. In either case, the successor must receive a new independent source audit and source-free runtime calibration before any payload access.

## Exact producer bytes audited

| File | Bytes | SHA-256 |
|---|---:|---|
| `ARTIFACT_SHA256SUMS.txt` | 505 | `e19acb3c8e888dddb8ae296f05b7541ba9db47e017619aea0c9dd341f0c5b3f4` |
| `README.md` | 9820 | `95824df46b8650f74d003fef76877005b6f4e32ab99fafbfe4aa9749f5d3741e` |
| `free_order_oracle.py` | 28692 | `7329ee7cd6838e21db7ae81b9ee16843548c1f99297e6d007487c450ceaff820` |
| `protocol_lock.json` | 8690 | `71e45e62fa86238e89424f24ecf346cb2cd49715e569534c2ff817011015c66a` |
| `source_bindings.json` | 2878 | `3454b718a65efc02c32463f955c10ff393f4218fac04f358107960ff3735990d` |
| `test_source_only.py` | 7549 | `83424cff84195118975e31aabf725478cfab3c57b40f4f20fd702dd809c02288` |
| `verify_package.py` | 8893 | `5e1940b3c626706815c10febe6dc6e35a47d4e28548df9379915e5025c5c5499` |

Run the sealed audit verifier with `python -B verify_audit.py`. It reads only this audit set and the seven frozen producer source files in the sibling directory.
