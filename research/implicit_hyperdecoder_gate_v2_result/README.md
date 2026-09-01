# SILWARP-v2 auxiliary result evidence

This directory preserves the completed, preregistered SILWARP-v2 auxiliary
gate.  It is a negative result: both fixed training seeds selected the exact
identity bypass at updates 256 and 512, so the frozen update-512 rule returned
`HARD_KILL_FROZEN_SILWARP_CELL_AT_UPDATE_512`.

The run used CuPy 14.2.0 on the supplied RTX 5090.  Its source-free GPU
preflight opened no tensor payload.  During the run, only the frozen auxiliary
fit and calibration sets were decoded.  The result records
`confirmation_opened=false` and `pinned_panel.opened=false`.

This is not a codec checkpoint or a positive compression claim.  It closes
only the exact SILWARP-v2 ideal-channel training cell; work continues in other
architecture families.

The raw receipt, sentinel, result, append-only checkpoint metadata, and run log
are copied from the immutable RunPod paths.  Large optimizer state and negative
model blobs remain on the RunPod and are content-addressed by the metadata.
