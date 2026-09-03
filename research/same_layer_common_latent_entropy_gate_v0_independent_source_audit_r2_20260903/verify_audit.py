"""Stdlib-only closure verifier for repaired-source audit r2."""
from __future__ import annotations
import argparse, hashlib, json, stat
from pathlib import Path

PRODUCER_MANIFEST="b92d4b5f307ba1d2b6bc6370d0b7cd118c4ab138dc6c8943402efe632a2a5d8f"
PRODUCER_ROOT="f9fe8b64b31edc7599e8e9c302b7e283b2aed9cc24c165916ae3447a9f78311c"
def req(x,m):
    if not x: raise RuntimeError(m)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(root, expected):
    req(root.is_absolute() and root.is_dir() and not root.is_symlink(),"root")
    req(len(expected)==64 and all(c in "0123456789abcdef" for c in expected),"external digest")
    mp=root/"SOURCE_MANIFEST.json"; req(sha(mp)==expected,"manifest digest")
    raw=mp.read_bytes(); m=json.loads(raw)
    req(raw==(json.dumps(m,sort_keys=True,separators=(",",":"))+"\n").encode(),"canonical manifest")
    req(m["schema"]=="same_layer_common_latent_independent_audit_r2_manifest_v0","schema")
    rows=m["files"]; names=[r["name"] for r in rows]
    req(names==sorted(names) and len(names)==len(set(names)),"row order")
    req(sorted(p.name for p in root.iterdir())==sorted(names+["SOURCE_MANIFEST.json"]),"closure")
    canon=[]
    for row in rows:
        p=root/row["name"]
        req(set(row)=={"bytes","name","sha256"} and stat.S_ISREG(p.lstat().st_mode) and not p.is_symlink(),"row")
        req(p.stat().st_size==row["bytes"] and sha(p)==row["sha256"],"member")
        canon.append({"bytes":row["bytes"],"name":row["name"],"sha256":row["sha256"]})
    ar=hashlib.sha256(json.dumps(canon,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    req(ar==m["audit_root_sha256"],"audit root")
    r=json.loads((root/"AUDIT_RECEIPT.json").read_bytes())
    req(r["status"]=="PASS_REPAIRED_SOURCE_ELIGIBLE_FOR_SEPARATE_DEPLOYMENT_REVIEW","verdict")
    req(r["source_manifest_sha256"]==PRODUCER_MANIFEST and r["source_root_sha256"]==PRODUCER_ROOT,"producer pins")
    req(r["payload_accessed"] is False and r["source_files_modified"] is False,"boundary")
    x=r["full_function_regressions"]
    req(x["old_regression_closed"] is True and x["failure_eligible"] is False and x["controls_called_on_physical_failure"]==0,"regression")
    req(x["valid_survivor_controls_run"]==8,"survivor controls")
    f=r["feasible_rate_rule"]
    req(f["exact_endpoint_set"]==["2.15","2.5"] and f["required_true_predicates"]==["status","capacity_ok","strictly_below_2x"],"rate rule")
    c=json.loads((root/"CUPY_PARITY_RECEIPT.json").read_bytes())
    req(c["status"]=="PASS_SOURCE_FREE_CPU_CUPY_PARITY" and c["manifest_sha256"]==PRODUCER_MANIFEST and c["payload_accessed"] is False,"CuPy")
    return {"schema":"same_layer_common_latent_independent_audit_r2_verification_v0","status":"PASS_SEALED_REPAIRED_SOURCE_REVIEW","manifest_sha256":expected,"audit_root_sha256":ar,"producer_manifest_sha256":PRODUCER_MANIFEST,"verdict":r["status"]}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--package",required=True); p.add_argument("--manifest-sha256",required=True); a=p.parse_args()
    print(json.dumps(verify(Path(a.package).resolve(),a.manifest_sha256),sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
