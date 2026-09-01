# Polar nearest-codeword / list-search kill audit

Status: **hard-killed before GPU execution**.  This branch does not modify the
STRATA checkpoint or its decoder.

## Question

The frozen encoder selects each unfrozen polar input bit by posterior MAP
successive cancellation (SC).  A decoder accepts any legal information-bit
path, so a better encoder could in principle use list search or coordinate
refinement and transmit only the winning path.  No list identifier is needed.

The relevant question is not whether a different path sometimes helps.  It is
whether this operational search gap can plausibly supply the missing
`0.16096404744368117` Gaussian-equivalent bpw, while retaining exact causal
arithmetic length and the same physical rate.

At physical `R=2.5`, the Gaussian reference is `2^(-2R)=0.03125` and the goal
is `0.025`.  The expert-local allocation projected `0.03090139432980219`, so
the remaining reduction from that design point is `19.0975017723%`, or
`0.15287192093031612` equivalent bpw.

## Evidence 1: all frozen real-Qwen STRATA-v2 blocks

[`prior_artifact_envelope.py`](prior_artifact_envelope.py) is source-free.  It
reads the 14 committed encoder metadata files from the independently decoded
28,311,552-weight STRATA-v2 panel and recomputes the operational excess over
both nominal test-channel distortion and the Gaussian distortion at each
block's exact screen rate.

| Quantity | Result |
|---|---:|
| weighted measured MSE / nominal test-channel D | `1.0102127572971418` |
| block range, measured MSE / nominal D | `1.0097106973`–`1.0124039644` |
| weighted measured MSE / exact-rate Gaussian MSE | `1.010895330695289` |
| full nominal-D gap closure | `1.0109511312%` MSE / `0.00732958291` bpw |
| full exact-rate Gaussian gap closure | `1.0777901890%` MSE / `0.00781681331` bpw |
| required gain / observed nominal closure | `20.8568x` |
| projected MSE after granting full exact-rate closure | `0.0305683421334515` |

This is deliberately generous: it grants a hypothetical search oracle the
entire observed finite-code loss.  It still misses `0.025` by a wide margin.
The envelope is empirical rather than a fixed-realization converse; the
per-block metadata are nevertheless the closest operational evidence because
they use the exact large-N code, Qwen staging distribution, causal model, and
arithmetic stream.

The committed audit is
[`prior_artifact_envelope.json`](prior_artifact_envelope.json).  It binds the
release manifest (`dcf87419...f55bd4`), independent decode audit
(`310d0435...7d78`), every block metadata hash, and its generating script.

## Evidence 2: exact small-N nearest-codeword stress test

[`small_n_rate_matched_probe.py`](small_n_rate_matched_probe.py) is a
CPU-only NumPy translation bound to the frozen encoder and Q31 BEC
construction hashes.  It uses normalized procedural Gaussian sources and
never imports CuPy or opens model weights.

At `N=16`, the relevant construction has only two constrained low bitplanes.
The probe enumerates every legal low-plane assignment.  Since bitplanes 3–6
are full-rate, it then chooses the nearest upper-plane reconstruction at every
coordinate.  The resulting unrestricted nearest-codeword SSE is therefore
**exact** for this small code, even though rate is ignored in that view.

For the rate-matched view, the probe sweeps 65 Gaussian-centered Lagrange
prices over the full-rate upper planes.  Every improving reconstruction is
then replayed through all six forced SC paths, using the literal causal prior,
Q16 frequencies, and 32-bit arithmetic coder.  A candidate is admitted only
when its native arithmetic length is no longer than the MAP-SC control.  It is
zero-extended to the control length and decoded again.  The bitstream itself
identifies the candidate: selector cost is exactly zero.

| Profile represented | Trials | Exhaustive constrained bits | Exact unrestricted nearest gain | Exact-length family gain |
|---|---:|---:|---:|---:|
| `D=0.04467541682474906` (`r=2.2421875`) | 32 | 6 | `12.6597763247%` | `4.4309106470%` |
| `D=0.03125` (`r=2.5`) | 64 | 9 | `8.0438391871%` | `1.5340183505%` |
| `D=0.023075408530304677` (`r=2.71875`) | 32 | 12 | `7.2083657241%` | `1.4868166610%` |

These are aggregate-SSE results, matching the panel objective.  Individual
`N=16` trials can fluctuate much more; that is precisely why the aggregate
and the large-N frozen evidence are the relevant statistics.  Even the exact,
rate-ignored small-code oracle stays below the required 20% aggregate gain.
At production `N=2^20`/`2^21`, the directly observed gap is only about 1%.

Result files:

- [`small_n_lowrate_result.json`](small_n_lowrate_result.json)
  (`78684232...184d6`)
- [`small_n_rate_matched_result.json`](small_n_rate_matched_result.json)
  (`67994ded...0d8aa`)
- [`small_n_highrate_result.json`](small_n_highrate_result.json)
  (`eab5628b...b443b`)

## Qwen residual cross-check

The independent residual marginal audit on the complete frozen v2 panel is
also negative for a hidden search opportunity:

- moment-matched GGD entropy-power / variance: `0.9999999484302325`;
- excess kurtosis: `0.0010131005845259722`;
- 2048-value group variance GM/AM: `0.9995069551621844`;
- maximum tested absolute lag correlation: `0.00032147690416820695`;
- cross-fit histogram net gain after its table: `-0.001253937482248183` bpw.

The audit file is
`/workspace/INT2__compression/residual_marginal_audit_v2.json`, SHA-256
`bae09f440c591a0bca2d6be7410e5d6f61e355cc05a68943d81a39a4b82167e5`.
Those statistics are not used as a mathematical converse; they rule out the
specific hypothesis that SC search is concealing a large non-Gaussian or
short-memory residual structure.

## Decision

Do not spend GPU time on full-block SCL, beam search, or coordinate refinement
for the present serialized polar code.  Promotion would require a new
source-free argument showing a qualitatively different search gap above
`0.15` bpw.  The measured large-N opportunity is `0.0073`–`0.0078` bpw, the
exact small-code rate-matched opportunity is `1.49%`–`4.43%`, and even the
rate-ignored small-code nearest oracle is only `7.21%`–`12.66%`.

This conclusion is an evidence-based engineering kill, not a theorem that SC
suboptimality is monotone in block length.  The combination of exact small-N
search, literal large-N Qwen streams, and near-Gaussian residual audit is the
reason a production GPU pilot is not justified.

## Reproduction

From the repository root, with any Python containing NumPy:

```powershell
python research/polar_search_gap/prior_artifact_envelope.py `
  --output research/polar_search_gap/prior_artifact_envelope.json

python research/polar_search_gap/small_n_rate_matched_probe.py `
  --distortion 0.03125 --trials 64 --lambda-points 65 `
  --output research/polar_search_gap/small_n_rate_matched_result.json
```

Use `--distortion 0.04467541682474906 --trials 32` and
`--distortion 0.023075408530304677 --trials 32` for the endpoint stress tests.
All result files record upstream/script hashes, Python/NumPy versions, runtime,
per-trial source and reconstruction hashes, native/equalized lengths, and
zero-extension arithmetic round trips.
