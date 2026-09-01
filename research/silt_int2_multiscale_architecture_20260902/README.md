# SILT-INT2: a scaffold/seriation-induced lifting tree for legal INT2 labels

Date: 2026-09-02  
Status: source-free architecture assessment; **not** a codec result and **not**
authorization to open a model, current-container, or Gaussian-control payload.

This report uses only the universal codec contract, published aggregate
repository results, algebra, and primary literature. It does not inspect a
Qwen tensor, a current compressed reservoir, or a matched-control payload.

## Executive verdict

The ordinary interpretation of graph lifting is already a dead end here. A
real-valued Haar/tree predictor is a special case of predicting one
micro-neuron from another. The repository's stronger free-predecessor oracle
gave every neuron its best distinct predecessor and a full `3 x 3` role
regression, yet found only `s=0.01446386 bpw` before side cost, versus roughly
`0.18580705 bpw` required by that family. A graph chosen to make raw weight
differences small cannot plausibly close this target.

There is, however, a distinct multiscale hypothesis that the predecessor test
does not contain:

> A charged permutation plus an exactly reversible finite-field lifting tree
> may turn non-local, high-order dependencies among **legal quantizer labels**
> into a low-complexity tree process, even when every real-valued pairwise
> regression and covariance is weak.

The proposed architecture is **SILT-INT2**:

```text
canonical SwiGLU role orientation
    -> legal PTQ / SC decision tensor
    -> charged balanced micro-neuron (or transform-lane) seriation
    -> bijective GF(2) / Z4 lifting circuit
    -> small nonnegative unifilar tree automaton
    -> 256-way interleaved exact Q16 rANS
    -> one page-contiguous frame per expert
```

The lift is entropy-neutral in principle: it is a bijection. Its purpose is to
make the true joint law cheap for a bounded decoder, not to claim that a
transform creates information. This gives the branch a decisive early test.
If a deliberately favorable, free-tree oracle cannot save at least
`0.15288997 bpw` on the exact same reconstruction, stop. A finite candidate
should show at least about `0.17 bpw` gross saving before engineering.

My recommendation is **promote only to a sealed source-free mechanism test and
then a free-tree held-out entropy oracle**. Do not yet build a full Qwen codec.
The branch is scientifically distinct and cheap to falsify, but the prior odds
are moderate-to-low because so many simpler dependency models have failed.

## 1. Exact opportunity and non-negotiable target

The current independently decoded aggregate result is:

```text
R0 = 2.5 bpw
D0 = 0.030902167403153148
F0 = D0 * 2^(2 R0) = 0.9888693569009007
s0 = -0.5 log2(F0) = 0.008074080480766676 bpw
```

The final target `F <= 0.8` is equivalent to

```text
s_target = -0.5 log2(0.8) = 0.16096404744368115 bpw.
```

Keeping the reconstruction exactly unchanged therefore requires net physical
saving

```text
Delta R_net = s_target - s0
            = 0.15288996696291447 bpw,

R_same_D_max = 2.3471100330370855 bpw.
```

For the published six-expert planning geometry
`E=6, m=768, h=2048`, the source count is

```text
N = 3 E m h = 28,311,552 weights,
required net saving = 4,328,552.25 bits = 541,069.03 bytes.
```

The same-rate distortion ceilings are:

| Physical rate | Maximum relative MSE for `F=0.8` |
|---:|---:|
| 2.15 | 0.04061261981781178 |
| 2.3471100330 | 0.03090216740315314 |
| 2.50 | 0.025 |

SILT first takes the clean lossless-recode route: decode exactly the same legal
SC/INT2 decisions, hence preserve `D0`, and ask whether a better joint law can
put those decisions below `2.347110033 bpw`. Only a near miss earns a joint
lossy analysis-by-synthesis stage.

## 2. What the signal-processing literature contributes—and what it does not

### 2.1 Lifting gives exact invertibility

The lifting scheme factors a perfect-reconstruction transform into triangular
predict/update maps and supports irregular domains
([Sweldens, 1998](https://doi.org/10.1137/S0036141095289051)). Integer-to-
integer lifting is established by
[Calderbank, Daubechies, Sweldens and Yeo](https://scholars.duke.edu/publication/763243).
Graph lifting extends the even/odd predict-update construction to arbitrary
graphs
([Narang and Ortega](https://eprints.lib.hokudai.ac.jp/dspace/bitstream/2115/39737/1/TA-P1-10.pdf)),
and critically sampled graph filterbanks can be made perfect reconstruction
([Narang and Ortega](https://arxiv.org/abs/1106.3693)). Adaptive lifting can
remain perfectly reconstructing when the adaptation is decoder reproducible
([Piella and Heijmans](https://doi.org/10.1109/TSP.2002.1011203)).

These results justify the mechanism, not a compression gain. A bijection
preserves joint entropy. Any rate gain must be a reduction in finite model
mismatch after every tree and model byte is charged.

### 2.2 Graph wavelets and scattering are topology tools, not free codecs

Diffusion wavelets provide multiscale bases on graphs
([Coifman and Maggioni](https://doi.org/10.1016/j.acha.2006.04.004)), while
treelets learn a hierarchy and local orthogonal basis for unordered variables
([Lee, Nadler and Wasserman](https://stat.cmu.edu/~annlee/AOAS137.pdf)). Graph
scattering gives stable, permutation-aware multiscale summaries
([Gao, Wolf and Hirn](https://proceedings.mlr.press/v97/gao19e.html)).

Direct scattering is unsuitable for the raw-MSE codec: its modulus and pooling
are not invertible, and sending the lost phase/detail would restore the rate.
SILT permits a small fixed scattering signature only as an **encoder-side
candidate for choosing a transmitted tree**. The resulting permutation is
serialized; no descriptor is privileged decoder side information.

### 2.3 TTN/MERA suggests the topology, not a dense Born decoder

Tree tensor networks can model long-range dependencies with a hierarchy better
matched to non-chain data than an MPS
([Cheng et al.](https://arxiv.org/abs/1901.02217)). The wavelet/MERA connection
formalizes the role of local disentanglers followed by coarse graining
([Evenbly and White](https://arxiv.org/abs/1602.01166)). Learned lifting plus
tree entropy models is already effective in image compression
([Sahin and Kamisli](https://arxiv.org/abs/2212.03616)).

Those papers do not justify a dense Born-amplitude model here. A Born TTN needs
normalization/contraction machinery, floating-point probability semantics, and
larger state. SILT uses a nonnegative **unifilar tree automaton**: Q16
probabilities, deterministic decoded state, one table lookup per bit, and a
stack of logarithmic depth. It is a restricted operational TTN, not a quantum
claim.

### 2.4 Common/private trees are mathematically valid but cold-read hostile

Gray-Wyner common/private coding is classical
([Gray and Wyner](https://www.nokia.com/bell-labs/publications-and-media/publications/source-coding-for-a-simple-network/)),
and locally decodable source coding has explicit rate/locality tradeoffs
([Makhdoumi et al.](https://arxiv.org/abs/1308.5239)). But an ordinary balanced
Haar tree over `E` experts stores one root coefficient and one detail on each
root-to-leaf level. With comparable rate per coefficient field, decoding one
expert reads about `1+log2(E)` full fields. For `E=128`, that is about `8x`, not
`<2x`.

Shared levels can meet the read gate only if they are very low-dimensional or
procedurally generated. Existing shared-template and aligned-expert evidence is
strongly adverse. SILT therefore keeps the lifting tree **inside one expert**.
No weight-valued common stream is part of the frozen architecture.

### 2.5 Type classes are search shells, never entropy rebates

Enumerative source coding assigns an exact rank within a set
([Cover, 1973](https://doi.org/10.1109/TIT.1973.1054929)), and permutation
source codes provide structured vector codebooks with simple encoding
([Berger, Jelinek and Wolf](https://ntrs.nasa.gov/citations/19720034725)). These
ideas are useful only with honest accounting. For any deterministic statistic
`b=f(q)`,

```text
H(q) = H(b) + H(q | b).
```

Sending `b` and an enumerative rank cannot save entropy by itself. SILT uses a
type class only in the optional lossy stage: moves within a fixed, already
charged class search for lower source MSE at the same code length.

## 3. The architecture

### 3.1 Universal canonicalization

For an expert with intermediate width `m` and hidden width `h`, orient the
three matrices as

```text
G in R^(m x h), U in R^(m x h), D^T in R^(m x h)
```

and define canonical micro-neuron `j` as

```text
X_j = [G[j,:], U[j,:], D[:,j]] in R^(3h).
```

This uses only public shape and the semantic roles Gate, Up and Down. It does
not use a model name, layer number, expert number, router, activation, ancestor
checkpoint, or public reference weights.

The safest first integration is lossless: SILT receives the frozen front end's
legal binary SC decisions (or four-ary INT2 labels), plus the decoder-visible
route that maps decisions to role and transform lane. It changes only their
representation. If the current full RHT does not preserve the raw neuron axis,
the stage-0 wrapper uses the public transform-lane axis and makes **no
micro-neuron claim**. A final micro-neuron codec must rerun a separable
hidden-axis RHT that leaves `j` intact; wrapper and semantic-layout gains may
not be added.

### 3.2 A charged balanced tree

The encoder searches for a permutation `pi_e` of the `m` leaf lanes. Candidate
objectives are predeclared and small:

1. packed pairwise conditional label entropy;
2. a one-layer fixed Haar/scattering signature of the already produced label
   field;
3. direct held-out unifilar-tree codelength.

The chosen permutation is transmitted as a Lehmer/factoradic rank. The tree
topology is then the unique deterministic balanced binary tree over that leaf
order. Odd nodes are carried upward, so arbitrary positive `m` needs no fake
source weights and has exactly `m-1` internal nodes.

For `m=768`,

```text
log2(768!) = 6259.38 bits,
physical byte-rounded permutation = 6,264 bits = 783 bytes/expert.
```

An arbitrary source-derived tree would also need a topology description. SILT
does not take that liberty; balance is fixed by the format.

### 3.3 Exact finite-field lifting

At each internal tree node, pair left and right symbols `x,y`. For a binary SC
plane, operate over `GF(2)`:

```text
d = y XOR P(x,z)
c = x XOR U(d,z)
```

where `z` is decoder-visible ancestor/scaffold state. Decode exactly as

```text
x = c XOR U(d,z)
y = d XOR P(x,z).
```

For native four-ary labels, use the same triangular construction over `Z4`:

```text
d = y - P(x,z) mod 4
c = x + U(d,z) mod 4

x = c - U(d,z) mod 4
y = d + P(x,z) mod 4.
```

Any deterministic `P,U` give a bijection. The first finite library is the six
invertible `2 x 2` binary linear maps, requiring a 3-bit selector. One selector
per internal node is shared across roles and refinement planes:

```text
B_lift/e = 3(m-1) bits.
```

For `m=768`, this is 2,301 bits per expert. This tiny selector stream is still
physical and included in every ledger.

The lifting circuit emits one root aggregate and one detail per internal node.
It resembles a classical reversible MERA: local disentanglers expose XOR or
modular constraints before coarsening. It does **not** discard a coefficient,
change the legal reconstruction, or claim a Jacobian advantage.

### 3.4 Bounded-state tree probability law

Encode root-to-leaf/breadth-first coefficients with a nonnegative unifilar tree
automaton. A binary probability context contains only:

```text
SC/refinement plane:       6 values
semantic role:             3 values
tree-depth bucket:         4 values
decoded ancestor symbol:   2 values
unifilar state:            chi <= 16
left/right branch:         2 values
```

At `chi=16`, this gives `4,608` binary-frequency entries. One Q0.16 `uint16`
per context is 9,216 bytes. A deterministic `uint8` transition table needs at
most

```text
4 depth buckets * 2 branches * 16 states * 2 symbols = 256 bytes.
```

Together with backoff records, format constants, hashes and CRC, the complete
global model/header is capped at one 16 KiB page. If `chi=4` or `8` wins, its
smaller literal packet—not a hypothetical maximum—is charged.

The decoder stores at most `ceil(log2 m)` states per interleaved lane. For
`m=768, chi=16`, the logical causal stack is at most 10 four-bit states. There
is no dense `O(chi^3)` contraction, no floating-point normalization, and no
model/layer/expert identity context.

### 3.5 Exact physical entropy backend

Use integer Q16 rANS, not ideal NLL, with 256 interleaved lanes per expert.
ANS is an exact finite entropy-coding family with arithmetic-like rate and
small state
([Duda](https://arxiv.org/abs/1311.2540)). Each lane has a literal 32-bit final
state, so the state termination cost is

```text
256 * 4 = 1,024 bytes/expert.
```

The expert frame is page-contiguous:

```text
64-byte expert header
factoradic tree permutation
3-bit lifting selectors
256 rANS state words
rANS payloads with exact offsets
zero page tail
```

The only shared read is the 16 KiB probability-model/header packet. Decoding a
routed expert never reads another expert.

### 3.6 Optional entropy-constrained shell search

This stage is forbidden unless lossless SILT finds a substantial but
insufficient signal. For a legal quantizer codeword `q`, let `T_pi(q)` be its
bijective lifted representation and `L_theta` its exact finite rANS length.
Search

```text
q* = argmin_q [ ||x - xhat(q)||_2^2 + lambda L_theta(T_pi(q)) ].
```

On small blocks, exact enumeration or branch-and-bound gives the oracle. On
larger blocks, a move may swap labels or apply a cycle while preserving an
already charged histogram/type. The type vector and enumerative rank are
included in `L_theta`; only the lower MSE at fixed physical length is a gain.

No separately measured lossless saving and lossy improvement are added. The
stage emits one codeword, one reconstruction, one rate and one `F`.

## 4. Why this is not contained by earlier negative tests

| Earlier branch | Why it does not contain SILT | What it nevertheless warns us about |
|---|---|---|
| Randomized Hadamard / shape Hadamards | Linear real orthogonal mixing optimized Gaussianization; SILT is a charged data-adaptive tree and a nonlinear finite-field circuit over emitted labels. | A different real transform alone is very unlikely to help. |
| ICA/projection pursuit through dimension 64 | Linear low-dimensional independence; an XOR/parity law can have zero covariance and no useful independent component. SILT spans the full tree with reused bounded state. | The tree must win in operational bits, not kurtosis. |
| Local categorical/bitplane contexts | Fixed serialized neighborhoods and bounded suffixes. SILT pays to move related lanes together and supplies ancestor/sibling state at logarithmic distance. | If the free-tree oracle still gives only local-context-sized gain, kill. |
| TT/MPO/Kronecker/low-rank value fits | Approximate numerical weights. SILT represents a discrete joint law over legal labels; a full-rank value tensor can still obey label parity/type constraints. | Do not describe the lift itself as low-rank weight compression. |
| Free SwiGLU predecessor / neuron path | It strictly dominates unary **real-valued** graph prediction, but not high-order modular dependence with zero pairwise regression. | This kills continuous Haar compaction as SILT's source of gain. |
| ResMoE | Barycentre/common-expert plus residual across experts ([primary paper](https://arxiv.org/abs/2503.06881)). SILT has no weight-valued common expert and stays within one routed frame. | Cross-expert common pages are low priority. |
| CAMERA | Couples Gate/Up/Down micro-experts for pruning and mixed precision ([primary paper](https://arxiv.org/abs/2508.02322)). SILT codes the multiscale joint label copula and optimizes raw source MSE. | Merely concatenating three roles is not novel or sufficient. |
| Sequential WFA/MPS census | A chain law on one order. SILT pays for a balanced source-adaptive tree, applies reversible local disentanglers, and has logarithmic causal cones and parallel subtrees. | Both test the same ultimate source-information hypothesis; their gains cannot be added without one nested packet. |

The strongest novelty claim supportable today is narrow: I did not identify a
published weight-only PTQ codec combining a physically charged SwiGLU
micro-neuron seriation, reversible finite-field lifting of legal INT2/SC
labels, and a bounded exact tree-automaton/rANS law under routed-page
accounting. This is a research prior-art observation, not a patentability
opinion.

## 5. Exact planning ledgers

### 5.1 Generic formulas

For `E` experts of shape `(m,h)` and `N=3Emh` weights:

```text
B_perm   = E * 8 ceil(ceil(log2(m!))/8) bits
B_lift   = E * 3(m-1) bits
B_model  <= 16,384 * 8 bits
B_header = E * 64 * 8 bits
B_rANS   = E * 256 * 4 * 8 bits
B_tail   < E * 4,096 * 8 bits
```

All of these terms, including zero page tail, are charged in physical rate.
The model cap is not a free reservation: actual container bytes define `R`.

### 5.2 Six-expert `768 x 2048` planning instance

| New SILT state | Bits | bpw |
|---|---:|---:|
| Six byte-rounded factoradic permutations | 37,584 | 0.00132750 |
| Six sets of 3-bit internal-node lifts | 13,806 | 0.00048764 |
| One 16 KiB model/header page | 131,072 | 0.00462963 |
| Six 64-byte frame headers | 3,072 | 0.00010851 |
| 256 32-bit rANS states per expert | 49,152 | 0.00173611 |
| **Fixed subtotal before page tail** | **234,686** | **0.00828941** |
| Worst possible six frame tails | <196,608 | <0.00694444 |
| **Worst planning overhead** | **<431,294** | **<0.01523385** |

Using the exact `<4096` tail bound (`4095` bytes per expert) gives
`0.01523216 bpw`. Consequently:

```text
gross symbol-law saving needed, no page tail  = 0.16117937 bpw
gross symbol-law saving needed, worst tail    = 0.16812212 bpw
```

The report therefore uses `0.17 bpw` as the finite promotion line. Actual
padding and literal container size—not this planning bound—decide acceptance.

### 5.3 Cold routed read

At total `R=2.347110033 bpw`, the fair compressed share of one planning expert
is

```text
3*m*h*R/8 = 1,384,381.83 bytes.
```

If the total stream has six equal expert frames plus a 16 KiB global model,
the page-rounded planning union is

```text
1,384,448 expert-frame bytes + 16,384 shared bytes,
cold amplification = 1.01188269x.
```

As a deliberately conservative bridge, scale the currently reported
`1.16944444x` absolute read from `2.5` to `2.34711 bpw` and add a fresh 16 KiB
page. That upper planning estimate is about `1.25746x`. Both are below `2x`.
Neither is a result; the final verifier must enumerate the literal page union.

The architecture deliberately declines an expert-level common/private tree.
At uniform rate its `1+log2(E)` root path would violate the cold budget even
though its total storage were critically sampled.

### 5.4 Compute and transient state

For one `m=768,h=2048` expert:

* A full binary lift visits `m-1` pairs over `3h` symbols:
  `3h(m-1)=4,712,448` pair-symbol operations per bitplane. Two binary planes
  are about 9.42 million small integer operations.
* rANS performs about one exact table/state step per coded decision. At two
  decisions per weight this is about 9.44 million decisions.
* A packed all-pairs label distance screen costs approximately
  `m(m-1)3hb/(2*64)` 64-bit XOR/popcount words for `b` planes: about 28.3
  million at `b=2`, or 84.8 million at `b=6`. This is encoder-only and maps
  naturally to CuPy.
* The logical causal stack is `O(256 log m)` tiny states; the 16 KiB table and
  less than 2 KiB tree metadata fit in cache. The decoded expert itself
  dominates transient storage.

The sequential rANS dependency is the compute risk. The 256 streams are not a
decorative choice: they provide enough independent lanes for a GPU decoder,
at a measured `0.00173611 bpw` state cost. Production promotion requires an
RTX 5090 CuPy benchmark reporting end-to-end compressed bytes read, decoded
weights/s, joules/expert if available, and break-even tokens per expert load.
No bandwidth win should be called a power win before that measurement.

## 6. Ruthless staged oracle

### Gate 0 — source-free mechanism closure

Build fixtures with arbitrary legal `(m,h)` including odd `m`, three roles,
and both binary and Z4 alphabets:

1. tree-local XOR/parity constraints invisible to every suffix of depth 25;
2. iid labels with identical scalar histograms;
3. random leaf permutation, role transpose, malformed factoradic ranks, bad
   Q16 totals, truncated rANS state, and nonzero page-tail attacks.

Require independent standard-library decode and re-encode, exact label
identity, CPU/CuPy lift equality, and no gain on the iid control after model
bytes. Failure blocks any payload run.

### Gate 1 — favorable free-tree opportunity bound

On frozen auxiliary/held-out splits, grant the candidate all of the following
for free:

* best balanced leaf order from the predeclared search;
* best one of six invertible lifts at every internal node;
* the frozen `chi<=16` tree-law probabilities;
* ideal NLL rather than finite rANS termination.

Score the exact same legal reconstruction. Compare to the current prior on the
same symbol stream, not to raw two-bit storage.

```text
gross saving < 0.15288997 bpw: hard kill
0.15288997 <= saving < 0.17: pause; finite overhead almost certainly kills it
saving >= 0.17: authorize the physical tree/model cell
```

Because the oracle grants side information and ideal coding, a miss is a valid
family kill; a pass is only permission to continue.

### Gate 2 — topology ablation and containment

Run, on the same folds and bytes:

1. identity order + factorized law;
2. identity order + current sequential WFA survivor;
3. charged permutation + no lift;
4. charged permutation + lift + factorized law;
5. charged permutation + lift + tree automaton;
6. random balanced trees and reversal controls.

This identifies whether the gain comes from order, the bijection, or the tree
state. It also prevents relabeling a sequential-WFA gain as a graph result.
Promote only the complete nested row; never sum ablation deltas.

### Gate 3 — complete controls and whole-group uncertainty

For every candidate-selection step, rerun the entire pipeline on:

* moment-matched iid Gaussian sources passed through the identical PTQ;
* independent neuron/lane permutations within each hidden coordinate;
* role shuffles that preserve per-role marginals;
* tree-edge rewires preserving depth and degree.

Model selection must be repeated for every control, not merely refit at the
source winner. Confidence intervals and delete-group errors use whole experts
and whole layers, never millions of labels as iid observations. A universality
claim additionally needs an untouched SwiGLU-MoE family with different legal
dimensions.

### Gate 4 — literal physical recode

Serialize the factoradic tree, every lift selector, Q16 model, 256 rANS state
words, directories, checksums and page tails. From only those bytes:

1. independently reconstruct every original legal decision;
2. regenerate the exact prior reconstruction;
3. prove MSE equality bit-for-bit or rescore original-source BF16 MSE;
4. calculate `R=8B/N` and `F=D*2^(2R)` from literal bytes;
5. enumerate cold 4 KiB pages for every expert.

For the unchanged reconstruction the decisive condition is simply

```text
2.15 <= R <= 2.3471100330370855,
F <= 0.8,
max cold amplification < 2.
```

### Gate 5 — joint RD search only for a near miss

If physical lossless saving is below `0.10 bpw`, kill the architecture. If it
is between `0.10` and `0.15289 bpw`, run exact entropy-constrained search on
32-, 64- and 128-weight blocks under the same tree law. Require a held-out
ideal `F<=0.85`, leaving finite margin, and then retain at least 80% of the
gain in a literal packet. A type-class/Markov-basis walk is only a search
algorithm inside this gate; all statistics and ranks are charged.

### Gate 6 — sealed full result

The final artifact must bind source, source code, model, container, independent
decoder, reconstruction, physical byte ledger, page ledger and hostile tests.
Acceptance remains exactly:

```text
2.15 <= R <= 2.5,
F <= 0.8,
maximum routed cold read < 2x.
```

## 7. Principal risks and stop conditions

1. **No hidden entropy exists.** The most likely outcome is that the free-tree
   oracle saves far less than 0.153 bpw. Stop immediately.
2. **FOSP containment.** Any observed benefit explained by smaller real-valued
   neighbor residuals is already far too weak. SILT must show modular/high-order
   label gain after the value-prediction ablation.
3. **Tree overfit.** A source-adaptive permutation can memorize finite noise.
   Its exact rank is charged, and matched controls repeat the full search.
4. **RHT incompatibility.** If semantic micro-neuron alignment requires a new
   front end, the complete distortion and rate must be rerun. A transform-lane
   wrapper result cannot be transplanted.
5. **Model mismatch moves, entropy does not.** The lift is bijective. A gain
   against a weak factorized baseline that disappears against the current WFA
   is not a breakthrough.
6. **Sequential decode energy.** Even `1.01x` cold reads can lose in energy if
   rANS is slow. Measure, do not infer.
7. **Shared-table amortization.** Every model byte is charged over the literal
   evaluated artifact, not hypothetical future experts.

## 8. Final recommendation

SILT-INT2 is the strongest multiscale follow-on I can justify without
repeating a killed idea. Its radical element is not another wavelet basis; it
is the use of a **charged, source-adaptive tree of exact finite-field
disentanglers over legal quantizer labels**, paired with a small exact tree
law and an expert-local physical layout.

The architecture has three attractive properties:

* it can expose XOR/parity/type dependencies that covariance, ICA, unary
  predecessor regression and fixed local contexts provably can miss;
* it costs only about `0.00829 bpw` before page tail in the conservative
  six-expert planning case and has native cold read near `1.012x`; and
* it has a ruthless, inexpensive family-kill oracle before any finite codec or
  lossy search.

It also has one decisive weakness: nothing yet establishes that real SwiGLU
weights contain the required `~0.17 bpw` of tree-local label structure. The
correct next action is therefore a source-free synthetic implementation and
independent audit, followed—only if that passes—by the free-tree held-out
entropy gate. Ordinary continuous graph lifting, direct scattering, dense Born
TTNs, expert-level Gray-Wyner trees, and standalone type-class coding should
not receive compute before that gate survives.

