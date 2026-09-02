# MOSAIC-Q novelty-gap audit v0

Date: 2026-09-02

Scope: source-only repository and literature assessment. No Qwen/model,
STRATA/UWFA payload, completed-result payload, Gaussian-control payload, or
RunPod/CUDA resource was opened by this work. This file is not a numerical
compression result.

## Accounting boundary

The audited finite baseline is `R=2.5`, `D=0.030902167403153148`,
`F=0.9888693569009007`; the target is `D<=0.025`, or a `19.0995257%`
same-rate MSE reduction. The independently decoded TACTIC coarse object starts
at `R0=307/128=2.3984375`, `D0=0.036975150060595235`; a literal refinement to
2.5 bpw must therefore capture `32.3870222053737%` of its residual SSE. These
two capture thresholds are not interchangeable.

## Evidence matrix

| Hypothesis | Exact repository evidence | What is closed | What remains open | Literature boundary |
|---|---|---|---|---|
| Ramanujan / polyphase | `research/cyclo_fri4_normal_stage0_v0_qwen_run_20260901_5fb09e8/` plus its independent result audit: periods `{1,2,4}`, rank-four/free-tail dominant oracle, best charged `F=0.9379899307967997` at 2.5 bpw, source-energy capture `0.0019685047959490764`, side `0.007419444896556713 bpw`, maximum cold-read amplification `1.0084541062801933x`, decision `HARD_KILL_ABSOLUTE_DOMINANT_ORACLE_NO_CONTROLS`. `research/mosaic_secondary_oracles_v0/` and its independent source audit implement a public non-dyadic period bank but opened no payload and retain `HOLD_PRODUCTION_ADAPTER_SCORER_BACKEND_AND_IO_BINDING`. | The declared CYCLO-FRI4 normal-field family at periods 1, 2 and 4. It is not worth rerunning under another name. | Exact-period non-dyadic components `{3,5,...,127}` on the real coarse residual, native and coarse-derived orders. No Qwen/coarse result exists. | Ramanujan subspace pursuit is established signal processing, not itself novel: Deng and Han, [arXiv:1512.08112](https://arxiv.org/abs/1512.08112). |
| Hankel / annihilating filter / displacement rank | CYCLO-FRI4 is a narrow rank-four annihilator on a polar normal field. `research/mosaic_secondary_oracles_v0/` implements only binary16 AR orders `{1,2,4,8,12}` with finite-length inverse-noise pullback and an ideal iid-Gaussian innovation backend; source audit confirms mechanics only. `research/tactic_cage_graph_krylov_oracle_v0/SECONDARY_SCREENS.md` specifies, but does not execute, a `256 x 3841` coarse-seriated Hankel lift at ranks `{1,2,4,8,16,32}`. No `A_c,B_c` displacement-rank packet exists. | The tested CYCLO rank-four/period-1,2,4 family only. | Coarse-seriated Hankel rank, higher-order/long recurrences, and coarse-derived displacement operators are unmeasured. The capped AR result is also still open on payload. | Annihilating filters/FRI: Vetterli, Marziliano and Blu, [IEEE TSP 2002, DOI 10.1109/TSP.2002.1003065](https://bigwww.epfl.ch/publications/vetterli0201.html). Displacement rank: Kailath, Kung and Morf, [JMAA 1979, DOI 10.1016/0022-247X(79)90124-0](https://www.sciencedirect.com/science/article/pii/0022247X79901240). |
| Bispectral Volterra lifting | `research/tactic_cage_graph_krylov_oracle_v0/design_lock.json`, `SECONDARY_SCREENS.md`, and `secondary_hooks.py` explicitly defer this branch; no implementation or result exists. `research/neural_flow_oracle/` hard-kills the listed affine/MLP conditional-density families (`-0.00276119065423034 bpw` control-adjusted for the MLP), but those models are not a fixed bicoherence-selected quadratic/cubic perfect-reconstruction lift. | Only the listed long-context affine-flow and bounded MLP families, not phase-coupled Volterra lifting. | Sparse, auxiliary-frozen lag interactions in an exact lift, with a phase-randomized equal-spectrum control. | Bispectral nonlinearity testing: Hinich, [JTSA 1982, DOI 10.1111/j.1467-9892.1982.tb00339.x](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-9892.1982.tb00339.x). Nonlinear perfect-reconstruction lifting predates this application: Claypoole et al., [IEEE TIP 2003, DOI 10.1109/TIP.2003.817237](https://pubmed.ncbi.nlm.nih.gov/18244701/). |
| Sheaf / Hodge | A repository-wide search finds no sheaf codec, sheaf Laplacian, restriction-map oracle, or Qwen result. `research/tactic_cage_graph_krylov_oracle_v0_qwen_result_20260902_4d5220fe/` hard-kills one scalar coarse-path DCT/Krylov family: nominal waterfill gain `0.30344543088152887 bpw`, but control `0.30328819836395404`, leaving only `0.00015723251757482348 bpw` source-specific excess. Scalar graph spectra and whole-matrix manifold screens do not contain heterogeneous Gate/Up/Down stalks and role-dependent restriction maps. | The declared scalar coarse-path graph/Krylov family. | A role-heterogeneous sheaf global-section oracle is entirely unimplemented. Give maps away only for a first containment gate; they must later be shared/frozen or charged. | Cellular sheaf Hodge Laplacians: Hansen and Ghrist, [arXiv:1808.01513](https://arxiv.org/abs/1808.01513). This establishes the mathematics, not a weight-PTQ result. |
| GF(2) and 2-adic recurrence | `research/mosaic_secondary_oracles_v0/` has exact two-plane four-label BM mechanics, but its audit states that direct aliasing to current STRATA is invalid because STRATA reconstructs 64 indices from six complete level-major polar passes. `research/strata_sc_gf2_recurrence_adapter_v1/` targets those six selected-SC segments and retains every scale/state field, but is source-only and currently `PENDING_FINAL_10_TEST_RERUN`; no payload result exists. Its pre-payload bound hard-kills raw decisions (`5.17308016176577 bpw`) while a zero-complexity floor (`0.700443974247685 bpw`) survives. A 2.5-bpw recurrence packet needs aggregate BM complexity at most `25,474,112` before omitted costs. Its 128-bit owner mask is explicitly Qwen-shaped, not universal. No 2-adic/FCSR implementation exists. | Raw exact selected-decision storage at the cap. The abstract four-label serializer is not current-STRATA evidence. | Exact per-chunk GF(2) recurrence on authenticated SC decisions remains unrun; exceptions, streaming/multisequence recurrence and all FCSR/2-adic variants remain open. | Shortest GF(2) LFSR synthesis: Massey, [IEEE TIT 1969, DOI 10.1109/TIT.1969.1054260](https://crypto.stanford.edu/~mironov/cs359/massey.pdf). FCSR/2-adic synthesis: Klapper and Goresky, [Journal of Cryptology 1997, DOI 10.1007/s001459900024](https://www.cs.engr.uky.edu/~klapper/pdf/fcsr.pdf). |
| BM3D / coarse collaborative coding | `research/cross_expert_patch_dictionary_superoracle_v1/result.json` reports optimistic two-atom capture `0.14929587104823525` and `+3SE=0.1494499134316688` versus required `0.2100008524`, but its independent audit is `BLOCK_DO_NOT_EXECUTE_OR_USE_AS_CEIPA_HARD_KILL`: it covers only cross-expert raw Up/Down patches and is not a superset of a coordinate-conditioned decoder. The coarse graph/Krylov null is adverse evidence for coarse signatures but not a converse for nonlocal matched groups. `SECONDARY_SCREENS.md` freezes a causal coarse-signature patch screen; it has not run. | The literal two-atom cross-expert raw-patch diagnostic as a positive route; it cannot be promoted and cannot close CEIPA/BM3D. | Within-expert, coarse-only matching of residual patches followed by collaborative transform is open, but should not be built before its neighbor-correlation pretest passes. | BM3D grouping plus collaborative 3-D transform is established: Dabov et al., [IEEE TIP 2007, DOI 10.1109/TIP.2007.901238](https://researchportal.tuni.fi/en/publications/image-denoising-by-sparse-3-d-transform-domain-collaborative-filt/). |
| Posterior HMT / GSM | `research/uwfa_sc_posterior_centroid_v0_final_independent_audit_20260902/` passes source mechanics only and explicitly reports no Qwen/control result, no inference-ready routed posterior application, and no posterior head applied inside the routed session. `research/uwfa_sc_posterior_centroid_v1/` repairs the integer ABI but remains a one-page, block-occupancy affine head (`per_bin_centroid_table=false`), not an HMT/GSM. `research/epsilon_tcq_wfa_early_gate_v0/` source-free mechanics include state-conditioned binary16 centroids and a state-permutation control, but its independent audit is `HOLD_BINDINGS_REQUIRED_BEFORE_PAYLOAD` and no legal STRATA adapter or model result exists. Local evidence is adverse: decoded-affine best favourable transferred `F=0.998785937659889`; RAVEL-v1 captures only `0.00057309` and has oracle `F=1.01765194`. Neither is a nonlocal latent-tree posterior. | Small local affine/LUT correction families. | A graph/wavelet hidden Markov tree or Gaussian-scale-mixture posterior with full expert-local smoothing, literal model bytes, and routed application is untested. | HMT wavelet dependencies: Crouse, Nowak and Baraniuk, [IEEE TSP 1998, DOI 10.1109/78.668544](https://repository.rice.edu/items/5b84d16a-255a-4f1c-8870-0840f5ace39c). GSM posterior means: Portilla et al., [IEEE TIP 2003, DOI 10.1109/TIP.2003.818640](https://www.cns.nyu.edu/pub/lcv/portilla03-reprint.pdf). |

## Cheapest genuinely new dominant oracle

Run the **non-dyadic Ramanujan coarse-residual gate** already source-frozen in
`research/mosaic_secondary_oracles_v0/`. GF(2) recurrence is already an active
source-frozen branch; a BM3D neighbor correlation or eight-lag bicoherence test
would be cheaper but is only a pretest, not a dominant oracle. The non-dyadic
Ramanujan projection is the cheapest unexecuted gate that gives a bounded
containment test for its complete frozen period bank and requires no learned
dense basis.

This dominance is family-scoped: it can close corrections in the frozen
Ramanujan span under its declared bit-allocation envelope, not arbitrary
periodic or nonlinear codecs.

### Frozen source experiment

1. Decode the authenticated `307/128`-bpw coarse packet once and buffer its
   reconstruction/symbol state. Form exact FP64 residuals in original source
   coordinates. No second compressed-expert read is allowed.
2. Split every role into canonical 4,096-value blocks with a public,
   shape-derived tail rule. Test the already frozen periods
   `{3,5,6,...,127}` in native order and one coarse-symbol-derived seriation.
   Ordering, period bank and QR convention are frozen before test layers.
3. Report separately: free-amplitude public-prefix projection; a literal
   fixed-prefix FP16 coefficient packet inside 384 bits; source-selected
   support with its combinatorial rank and FP16 amplitudes inside the same 384
   bits; and the ideal waterfill diagnostic. Score every reconstruction after
   exact inverse ordering in source coordinates.
4. Select ordering/rank rules only on disjoint whole layers. The Qwen pilot
   must have at least five untouched test layers; confidence is clustered by
   whole layer/expert, never by scalar weight.

### Absolute and control kill gate

Hard-kill before controls unless the dominant source oracle itself reaches
`D<=0.025` at a final literal rate no greater than 2.5 bpw. If it survives,
run both controls, repeating the *complete* rank/order/support selection:

- one frozen odd-affine phase-destroying permutation within every public
  block; and
- eight complete matched-Gaussian coarse pipelines using the frozen seeds
  `10619863, 10619881, 10619909, 10619927, 10619953, 10619971, 10619999,
  10620017`, matching each block's FP64 mean and centered energy before the
  same coarse encode/decode and Ramanujan search.

Let `g=-0.5*log2(D_candidate/D_coarse)` be the rate-equivalent residual gain.
Hard-kill unless the whole-owner 95% lower confidence bound of

`g_Qwen - max(g_phase_control, g_Gaussian_control)`

is at least `0.03 bpw`. Control subtraction cannot rescue an absolute
`D>0.025` miss. A finite build is authorized only if one *literal* 384-bit
variant, with support, FP16 values, selector, header, CRC and padding all
charged, still yields `F=D*2^(2R)<=0.8` for `2.15<=R<=2.5`.

### Physical-read kill gate

Emit one page-aligned expert-local container and independently decode it.
Define cold-read amplification as

`(unique expert pages + charged shared/global pages + every repeated page or
refetch) / equal-share pages at the emitted physical rate`.

Count external storage, host parse/scratch, transfer and accelerator HBM in
separate ledgers. Hard-kill if any routed expert needs a second compressed
packet pass, any uncharged model/basis/centroid page, or maximum cold-read
amplification `>=2`. A planned byte layout or `physical_bytes/physical_bytes`
ratio is not observed read evidence.

### Universal claim gate

A Qwen pass remains a pilot. Freeze the same period bank, shape/tail grammar,
selection rule and decoder, then repeat on a disjoint SwiGLU-MoE family. Model,
checkpoint, layer and expert identity may not drive the transform. Failure of
portable tails, variable expert cardinality, physical rate, `F<=0.8`, or the
read gate prevents a universal SwiGLU-MoE claim.
