# POLARIS-SC-v2 independent post-confirmation audit

Audit date: 2026-08-31  
Verdict: **PASS — no substantive discrepancy found**

This was a read-only recomputation after the preregistered confirmation was legitimately opened. No encoder or confirmation seed was rerun. The audit independently parsed the summary, all 32 encoder and decoder records, the raw reservoir, its directory and bit reservoir, the irreversible lock, frozen artifact hashes, and the whole-checkpoint rate ledger.

## Statistical recomputation

The registered family-wise rule is a one-sided 99% Bonferroni gate over three metrics:

- per-metric confidence: `1 - 0.01/3 = 0.9966666666666667`
- degrees of freedom: `31`
- Student-t critical value: `2.9080702125010807`
- UCB: `mean + tcrit * sample_sd(ddof=1) / sqrt(32)`

| Metric | Mean | Sample SD | UCB | Threshold | Margin | Result |
|---|---:|---:|---:|---:|---:|---|
| Logical arithmetic bits | 562736.8125 | 536.2487047882697 | 563012.4867203544 | 563464 | 451.5132796455873 | PASS |
| Absolute MSE | 0.052899761434925266 | 8.677231142727209e-05 | 0.052944369261643476 | 0.053304063510877964 | 0.0003596942492344879 | PASS |
| Sample-relative MSE | 0.05289955985234825 | 0.00016520577541300619 | 0.05298448867908174 | 0.053304063510877964 | 0.0003195748317962252 | PASS |

The absolute and relative UCBs are respectively 4.291463095268266% and 4.370491569901502% above the Gaussian limit `0.050765774772264724`, both within the registered 5% target. All 32 seeds are unique and their trial order exactly matches the frozen manifest. Three individual messages exceed the 563464-bit average allocation (maximum 564175 bits), which is permitted by the frozen global-pooling format; the aggregate fits.

## Decode and serialization checks

- All 32 independent decoder JSON records report success and the registered seed.
- A separate 452-check pass matched each decoder result to its encoder metadata, summary row, unpack audit, extracted record, payload, and recorded SHA-256 values.
- Maximum encoder-to-decoder MSE differences were `1.3877787807814457e-17` absolute and `2.7755575615628914e-17` sample-relative, far below `1e-12`.
- Every source payload's logical bits equal its raw bit slice in the reservoir; every independently extracted record reconstructs the same bits, big-endian u32 length, and raw FP16 scale.
- All encoder and decoder stderr logs are empty; their stdout transcripts are byte-identical to their JSON artifacts.
- The raw reservoir is exactly 2254144 bytes and has SHA-256 `ad0c35e72b5900ffa6ed353df1bf1b163d912b8bfb692fc8e5b318ea6f9eb3f5`.
- The reservoir contains 18007578 logical payload bits in an 18030848-bit fixed payload allocation, leaving exactly 23270 unused bits. Every unused bit is physically zero.
- Directory SHA-256: `562231b913d923ddebe29dc649f260db7e686abc872a5cd5b7a21b211e65c1b0`.
- Fixed-capacity payload SHA-256: `55fa245025b820032d32d328662a1aa46efe2b2c91702f12e62a0691a921409e`.

## Provenance and rate

- Harness SHA-256: `b9db626f716461d4932bffc52a85ba1c3e46ace70e8d9713eae5b6fcd37413b2`.
- Irreversible lock SHA-256: `401e21d2e64cd655f82f23b7a7aed21a5ec18d4928915868b841c936f7eb5693`; its mode is `0444` and its recorded harness and seed-list hashes agree with the summary.
- Frozen manifest SHA-256: `06967a4e852c9d39c97fe39b45d50df558471e0f35912d330b6dc1e7493df5e0`.
- Confirmation summary SHA-256: `f4988f8e92b99fa90a4fe2b6b153beb02a1cb123c49e21f64d90ce77f008e6b5`.
- Freeze, lock-open, reservoir, and summary timestamps occur in that order.
- Every initial and final frozen-artifact hash in the summary recomputes exactly.
- The whole-checkpoint ledger independently recomputes to 65641547920 bits / 8205193490 bytes / 2.1499176041040124 bpw for 30532122624 parameters.
- The exact rational cap is `floor(2.15 * P) = 65644063641` bits, leaving 2515721 nominal bits. The usable byte-aligned cap is 65644063640 bits, leaving 2515720 physical bits.

## Evidence boundary

The emitted 2254144-byte file is the 32-block Gaussian confirmation reservoir. The 8205193490-byte whole-checkpoint figure is a shape-recomputed, fixed-capacity serialization ledger/envelope; a full checkpoint bitstream of that size was not emitted by this confirmation run. This is an evidence-boundary clarification, not a discrepancy in the confirmation result.
