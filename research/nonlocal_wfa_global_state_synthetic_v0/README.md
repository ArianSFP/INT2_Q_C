# Nonlocal WFA global-state synthetic gate v0

Status: **source-free synthetic prototype only; no model-payload launch authority**.

This cell isolates the genuinely nonlocal claim behind an MPS/HMM entropy
census.  It uses a tied nonnegative weighted finite automaton (WFA), not a
finite-order suffix table.  The legal decoder context is only the public
coordinate phase inside a fixed 32-symbol reset block; no model, checkpoint,
layer, expert, router, activation, or source-weight identity is available.

## Exact model

For context `c`, symbol `y`, and hidden-state size `chi`, the expanded model is

```text
A[c,y] in R_+^(chi x chi)
p(y_t | y_<t,c_t) = alpha A[c,y_t] 1 /
                     alpha (A[c,0]+A[c,1]) 1.
```

Every serialized row satisfies

```text
sum_{y,j} A[c,y,i,j] = 65535
```

exactly.  The prototype is unifilar, so each `(c,y,i)` row has one nonzero
successor.  This is still a symbol-conditioned matrix WFA: its state is a
learned/selected global parity bank rather than the last `d` symbols.  A full
matrix representation costs

```text
256 + 2*chi + 2*C*2*chi^2 bytes,
```

while the exact sparse packet costs

```text
256 + 2*chi + C*2*chi + 2*C*chi bytes.
```

Here `C=12`.  At `chi=64`, the packet is 3,456 bytes and occupies one 4 KiB
global page.  The code emits a real 32-bit arithmetic stream and independently
decodes it.

## Why the fixture defeats suffix contexts

A block has 26 iid body bits and six checksum bits.  Check `j` is the XOR of
body coordinates congruent to `j mod 6`.  All 32 site marginals are fair.  At
checksum `j`, any suffix of depth at most 25 omits body bit `j`, which appears
in no other checksum.  Conditional on the suffix and earlier checks, checksum
`j` is therefore still fair.  The population codelength of every such suffix
model is exactly 1 bit/symbol.

A 64-state automaton stores all six running parities and has entropy
`26/32 = 0.8125` bit/symbol, a gross saving of `0.1875` bit/symbol.  Training
fits Q0.16 transition weights for seven frozen topology candidates (`k=0..6`)
and selects `k` on an untouched validation set after charging its model page.
An independently generated iid control has identical marginals and is refit
from scratch.

Run:

```powershell
python test_source_only.py
python run_synthetic.py --output synthetic_result_v0
```

The experiment opens no Qwen or other checkpoint, no production symbol stream,
and no CUDA context.  There is no GPU path in v0; any later GPU evaluator must
be implemented only with CuPy.

## Capacity sanity check

Across one cut, a nonnegative bond-`chi` factorization is a mixture through a
`chi`-valued common state, so

```text
I(left; right) <= log2(chi).
```

A Born-amplitude MPS has an effective squared bond and a roughly
`2 log2(chi)` cut-information ceiling.  For `chi=64`, these figures are 6 and
about 12 bits.  By contrast, the current standalone gap is
`0.15288996696 * 2048 = 313.118652...` bits per 2,048-symbol block.  One parity
constraint saves only one bit, or 0.00048828125 bpw at that block length.

This does **not** imply `chi >= 2^313`: a small state can be queried repeatedly,
and it can be reset and reused across local blocks.  This fixture reuses six
state bits over 64 independent 32-symbol blocks, producing 384 constraint bits
per 2,048 symbols.  The cut bound instead says that 313 *independent* bits all
crossing one common cut cannot pass through a small nonnegative bond.  A real
census should therefore report multiple cut locations and reset lengths, not
interpret `chi=64` as universally sufficient.

## Claim boundary

A negative production run of this frozen cell could close only its tied,
causal, Q0.16, reset-32, unifilar parity-bank family with `chi<=64` and the
specified train/holdout selection.  It could not close arbitrary HMMs, MPSs,
TTNs, non-unifilar state, longer resets, `chi>64`, Born models, learned
disentanglers, other universal public contexts, or a lossy rate-distortion
advantage.  Conversely, this synthetic pass says only that the architecture
can detect hidden global state.  It supplies no evidence that Qwen—or any
SwiGLU-MoE weights—contains the required structure.

