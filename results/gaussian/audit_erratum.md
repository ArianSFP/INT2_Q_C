# Erratum: POLARIS-SC-v2 post-confirmation audit endianness

Date: 2026-08-31  
Status: prose correction only; no frozen codec or confirmation artifact changed

This corrects one sentence in the hash-published report
`agent_polaris_sc_v2_postconfirmation_independent_audit_20260831.md`
(SHA-256 `fd1fea05c4ea57883daa535853e69e0e361a793677666698df50dfd2445c2541`).

The report incorrectly described the independently extracted staging record's
u32 length as big-endian. The actual, intentionally mixed framing is:

- reservoir directory entry: `u32be logical_length || raw f16le scale`;
- independently extracted staging record: `u32le logical_length || raw f16le scale || payload`.

The frozen unpacker converts the directory's big-endian length to the
little-endian staging header expected by the frozen block decoder. It copies
the two FP16 scale bytes verbatim. The reservoir directory, unpacked record
hashes, decoder results, MSE values, and PASS verdict are unaffected. The
hash-published JSON audit made no contrary endian claim.
