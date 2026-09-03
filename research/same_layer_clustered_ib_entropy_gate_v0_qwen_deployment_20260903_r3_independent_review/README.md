# Independent hostile review: r3 deployment

Disposition: **PASS for exactly one source-free RTX5090 parity preflight
attempt only**.

The exact r3 package authenticates at manifest SHA-256
`5bac35949d374961f280d12680e3755c8c0d45f58d1937d6fb19120547b3649f`
and source root
`ae1ae10c0a39b8167739498db13b69aaaf738a17b247ca7861c5d76a57d6c1ee`.
No r3 member was modified. No CuPy device, Qwen payload, RunPod, production
launcher, or network was accessed.

The production core, worker, and panel are byte-identical to r2. The changed
fixture is confined to the source-free runner, targeted regression, and test.
Production continues to derive its scale charge from authenticated quantizer
scales and does not import the fixture.

The independently reproduced group-size-2 regression uses exactly 256 scale
bytes/expert and yields favorable gain `0.6513992144263967` bpw. Its 5/2
endpoint is capacity-feasible (`320 >= 306` pages) and strictly below 2x with
maximum amplification `1.9651249492746525`. Thus the source/read survivor is
genuine and the eight-control branch is reachable. The 43/20 endpoint correctly
fails capacity.

All prior source-review blockers are closed. Full 16-expert, two-role,
2/4/8/16, eight-fold training and held-out assignment/count coverage remains
from r2; the r1 evaluator-schema defect remains absent; the production NumPy
wheel/native closure remains before its one-use claim; and the old scientific
core has not been substituted.

`AUTHORIZED_PREFLIGHT.json` grants one attempt from the exact fresh path it
names. Its sealed wrapper uses a fixed, shell-free argv and atomically creates a
persistent claim and exclusive stdout/stderr receipt paths before spawning the
source-free child. The first attempt consumes the authority even if it fails. There is no
retry authority and no Qwen, payload-read, network, `run_gate.py`, capability,
production, or scientific-result authority. Validate any emitted receipt with
`verify_preflight_receipt.py`.
