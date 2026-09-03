"""Stdlib closure verifier for the one-run deployment review."""
import argparse,hashlib,json,stat
from pathlib import Path
DEPLOY="a969382640ad69ee71b6029d901d7eade7b88112d582059d83b947e33d1767c3"
SOURCE="b92d4b5f307ba1d2b6bc6370d0b7cd118c4ab138dc6c8943402efe632a2a5d8f"
def req(x,m):
    if not x: raise RuntimeError(m)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(root,expected):
    req(root.is_absolute() and root.is_dir() and not root.is_symlink(),"root"); mp=root/"SOURCE_MANIFEST.json"; req(sha(mp)==expected,"manifest")
    raw=mp.read_bytes(); m=json.loads(raw); req(raw==(json.dumps(m,sort_keys=True,separators=(",",":"))+"\n").encode(),"canonical")
    req(m["schema"]=="same_layer_common_latent_deployment_review_manifest_v0","schema"); rows=m["files"]; names=[r["name"] for r in rows]
    req(names==sorted(names) and sorted(p.name for p in root.iterdir())==sorted(names+["SOURCE_MANIFEST.json"]),"closure"); canon=[]
    for row in rows:
        p=root/row["name"]; req(stat.S_ISREG(p.lstat().st_mode) and not p.is_symlink() and p.stat().st_size==row["bytes"] and sha(p)==row["sha256"],"member"); canon.append({"bytes":row["bytes"],"name":row["name"],"sha256":row["sha256"]})
    ar=hashlib.sha256(json.dumps(canon,sort_keys=True,separators=(",",":")).encode()).hexdigest(); req(ar==m["audit_root_sha256"],"root hash")
    r=json.loads((root/"AUDIT_RECEIPT.json").read_bytes()); req(r["status"]=="PASS_AUTHORIZE_EXACTLY_ONE_QWEN_RUN","verdict"); req(r["deployment_manifest_sha256"]==DEPLOY and r["parent_source_manifest_sha256"]==SOURCE,"pins")
    req(r["payload_accessed"] is False and r["deployment_files_modified"] is False and r["authorization"]["authorized_use_count"]==1,"boundary"); req(r["closure"]["all_other_member_bytes_equal"] is True and r["closure"]["sole_ast_literal_delta"]=={"deployment":True,"name":"PAYLOAD_EXECUTION_ENABLED","source":False},"delta"); req(r["internal_pins"]["all_exact"] is True,"internal pins")
    return {"schema":"same_layer_common_latent_deployment_review_verification_v0","status":"PASS_SEALED_ONE_RUN_DEPLOYMENT_REVIEW","manifest_sha256":expected,"audit_root_sha256":ar,"deployment_manifest_sha256":DEPLOY,"authorized_use_count":1}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--package",required=True); p.add_argument("--manifest-sha256",required=True); a=p.parse_args(); print(json.dumps(verify(Path(a.package).resolve(),a.manifest_sha256),sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
