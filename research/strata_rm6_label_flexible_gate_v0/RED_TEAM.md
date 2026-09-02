# Assumption red-team

1. **The 2.3232421875-bpw calculation is local-only.** Six RM(5,12)
   information sets contain 9,516 decisions per 4,096 coordinates. Current
   STRATA instead has 2^20/2^21 polar blocks and profile-dependent selected
   counts. Applying the local number to a global frozen-set swap is invalid.

2. **Information dimension is not arithmetic length.** The current causal
   arithmetic model can encode 9,516 decisions to fewer or more than 9,516
   bits. The literal packet charges its actual canonical length and rejects any
   packet above 1,280 bytes. There is no raw-bit fallback.

3. **Random frozen values hide polynomial appearance.** A zero-frozen RM
   plane is a low-degree coordinate function under the authenticated
   permutation. Current STRATA frozen values add a procedural random affine
   coset representative. Differences between codewords remain RM, but an
   observed random-coset label plane is not evidence of a low-degree function.

4. **Exact RM and truncated polar are different.** A selected count K defines
   RM(r,m) only when K is a complete RM dimension. Popcount-ranking an arbitrary
   K gives an RM-ordered truncated polar set, not an exact RM code.

5. **The local packet changes block topology.** It can concatenate 4,096-value
   index subblocks before the existing inverse global RHT/KLT, but that is a new
   container and SC reset schedule. The charged per-subblock scale and seed are
   conservative; a complete outer expert container is not implemented here.

6. **The GPU smoke is coordinate descent.** It evaluates exact 64-way costs and
   legal generator flips, but six successful flips do not approximate the
   global nearest joint RM6 codeword and give no Qwen performance evidence.

7. **The small-N oracle proves mechanics only.** Its 4,096-message enumeration
   is exact for N=8, not a production RM(5,12) decoder.

8. **The unconstrained 64-way result is a favorable lower bound.** It is not a
   legal RM packet and must never be reported as codec distortion.

9. **Cold-read and runtime claims remain open.** A 1x contiguous packet layout
   is plausible, but inverse transforms, directory traffic and materialization
   have not been integrated or benchmarked.

10. **No source evidence exists yet.** No Qwen, coarse-code or matched-control
    payload was opened. The correct payload verdict is HOLD.

11. **Sub-2.15 packets are not target passes.** Banks and source-free controls
    whose actual literal packet is below 2.15 bpw remain mechanism fixtures.
    A future refinement or padding field must be literal, charged and
    independently decoded before target promotion.

12. **Bank-0's dimension screen is not its emitted rate.** Its 9,516 selected
    decisions screen to 1,280 bytes if charged as raw bits, but the Q0.16 stream
    emits a data-dependent number of bits. Overflow above 1,280 bytes fails
    closed; it never falls back to raw selected decisions.
