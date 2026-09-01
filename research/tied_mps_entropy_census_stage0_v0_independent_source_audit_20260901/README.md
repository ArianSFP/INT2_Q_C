# Independent tied-HMM source audit

This directory is separate from the producer package. It never contains model
weights, decoded decision arrays, Gaussian controls, or a finite codec object.

`A469_BLOCK.json` permanently records the first advertised seal's closure
failure. A later producer reseal does not supersede or rewrite that receipt.
Any review of a later manifest is a new audit with its own exact manifest hash,
source-free evidence, and receipt.
