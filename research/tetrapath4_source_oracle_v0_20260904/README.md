# TETRAPATH-4 source-only dominant oracle

## Question answered

PAIRPATH-P2 tests a memoryless two-expert coupling.  Pairwise null results do
not exclude irreducible four-way synergy.  TETRAPATH-4 jointly moves the four
aligned labels

```text
(Up_e, Down_e, Up_f, Down_f) in {0,1,2,3}^4
```

and gives every tested probability family the same legal label choices, the
same deterministic symmetry-closed multistarts, and the same global rational
rate-distortion multiplier.

The compared families are:

1. four independent quaternary marginals;
2. all three exhaustive 2+2 pair factorizations;
3. the best of all 16 labelled Chow–Liu trees;
4. the pairwise maximum-entropy law matching all six one/two-way marginals,
   but containing no explicit third/fourth-order factor;
5. a sparse unary-plus-four-way-Gray-parity log-linear law fitted by IPF;
6. three `TETRAPATH-FIBER` laws in which a public Gray-bit or GF(4)-style
   pair map obeys `z=f(Up_e,Down_e)=f(Up_f,Down_f)`;
7. a full empirical 256-state law.

The source geometry is explicitly
`[Up_e, Down_e_transposed, Up_f, Down_f_transposed]`.  Down is required to be
transposed into the Up coordinate system before alignment; the helper rejects
shape or dtype ambiguity.

All source-derived tables, family selection, and convexified time sharing are
free.  This makes the result an intentionally favourable **kill-only** upper
envelope.  It cannot establish a finite codec.

## Score and gates

Each point minimizes one literal objective

```text
relative_MSE + lambda * rate_bpw
```

where `lambda` is one exact rational value shared by all four weights and every
family.  Frontiers are convexified before both comparisons:

- at equal physical-symbol rate, distortion improvement is converted to
  `0.5*log2(D_base/D_joint)` bpw;
- at equal MSE, the direct rate saving is `R_base-R_joint`.

The optimistic maximum is reported, with distinct fields for full joint versus
independent (codec relevance), versus best 2+2, versus Chow–Liu, and versus the
pairwise-maximum-entropy surrogate (residual connected information).  The
principal `G4` baseline is the lower envelope of independent, all pairings,
and Chow–Liu.  Sparse parity is also reported separately because it is an
inexpensive explicit four-way model.

- hard kill: `< 0.045 bpw`;
- standalone Up/Down target: `0.22933495044437174 bpw`;
- engineering margin: `0.27 bpw`.

The XOR fixture has exactly zero pairwise mutual information and exactly
`0.25 bpw` full-joint advantage, demonstrating why this experiment is not
subsumed by a pair census.  The pairwise-maximum-entropy surrogate also remains
uniform, isolating the full `0.25 bpw` as connected higher-order information.
A perfectly balanced IID fixture has exactly zero advantage and hard-kills.

The same XOR fixture supplies a constructive locality KAT.  With
`z=a xor b=c xor d`, the common stream costs one bit per tetrad and each expert
private stream costs one bit.  Total rate is three bits per tetrad, while a
routed expert touches common plus its private bit.  Relative to equal ownership
of the three-bit packet this is `2/(3/2)=4/3` ideal logical read amplification.
This is not claimed as a finite result: no interleaved 256-symbol stream is
treated as local, and real page rounding remains to be built and charged.

## Scientific limits

This oracle is coordinate-memoryless.  It does not test multiscale RENORM-Q,
cochain plaquettes, path transitions, the real six-plane deployed codec, or a
finite entropy packet.  A negative result closes only these bounded
memoryless four-variable families.  A positive result must survive held-out
whole-layer controls, charged model bytes, a literal independent decode, and
the routed-read constraint in a different audited package.

## Run the source-only tests

From the repository root:

```powershell
python -B research\tetrapath4_source_oracle_v0_20260904\verify_source.py `
  --package research\tetrapath4_source_oracle_v0_20260904 --self-test
```

The package is deliberately unsealed and grants no model, GPU, payload,
deployment, or execution authority.
