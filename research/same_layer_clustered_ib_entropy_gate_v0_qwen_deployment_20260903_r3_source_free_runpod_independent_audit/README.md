# Independent audit of the CBIB-1 r3 source-free RunPod preflight

This package independently authenticates the frozen r3 deployment and review
closures and validates the exact four copied remote evidence files without
replaying the one-use preflight. It makes no network, Qwen, payload, GPU, or
production access.

The disposition is `PASS_AUTHORIZED_SINGLE_PREFLIGHT_RECEIPT_INDEPENDENTLY_AUDITED`.
The receipt is a source-free CPU/CuPy parity result on an RTX 5090. It confirms
the exact 16-expert/two-role fixture, all four group sizes and eight folds, all
240 fitted models, exact training and held-out assignments and reconstructed
counts, full recursive gate parity, all eight controls, and the genuine
group-size-2 5/2-bpw source/read survivor. It is not a Qwen scientific result.

The authority's single preflight attempt is consumed. The copied evidence has
exactly four members; stderr is empty; the wrapper PASS binds receipt SHA-256
`38a4eb497983aa8b5a559fa96fcbcbb11dc77cdd78dabfff9d2d4d06c5bf1913`.
The authenticated wrapper source creates its persistent O_EXCL claim before
validation and child spawn and never invokes `run_gate.py`.

Run the detached validator with the four explicit package arguments shown by
`verify_evidence.py --help`. The external deployment and review manifest pins
are mandatory; no implicit current-directory trust is accepted.

No retry of the source-free preflight is authorized. Qwen execution requires a
separate, fresh, exact one-use production authority.
