#!/usr/bin/env python3
"""Result-only standard-library audit of the same-layer Up/Down super-oracle."""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, stat, sys
from pathlib import Path, PurePosixPath

MANIFEST="AUDIT_MANIFEST.json"
FILES={"README.md","audit_receipt.json","verify_audit.py",MANIFEST,"evidence/PACKAGE_MANIFEST.json",
       "evidence/design_lock.json","evidence/runpod_result.json","evidence/same_layer_alignment_oracle.py"}
HEX64=re.compile(r"^[0-9a-f]{64}$")
EXPERTS=(0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120); ROLES=("up","down")
class Checks:
    def __init__(self): self.count=0
    def require(self,x,label):
        self.count+=1
        if not x: raise AssertionError(f"check {self.count} failed: {label}")
def digest(x): return hashlib.sha256(x).hexdigest()
def canonical(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def strict_json(raw):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError("duplicate JSON key: "+k)
            out[k]=v
        return out
    def finite(x):
        v=float(x)
        if not math.isfinite(v): raise ValueError("nonfinite JSON")
        return v
    def constant(x): raise ValueError("nonfinite JSON: "+x)
    return json.loads(raw.decode("utf-8"),object_pairs_hook=pairs,parse_float=finite,parse_constant=constant)
def held_read(path):
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink(): raise ValueError("nonregular/link")
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_BINARY",0)|getattr(os,"O_NOFOLLOW",0))
    try:
        opened=os.fstat(fd); chunks=[]
        if (before.st_dev,before.st_ino,before.st_size)!=(opened.st_dev,opened.st_ino,opened.st_size): raise ValueError("identity changed")
        while True:
            b=os.read(fd,1<<20)
            if not b: break
            chunks.append(b)
        after=os.fstat(fd); raw=b"".join(chunks)
        if (opened.st_size,opened.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or len(raw)!=opened.st_size: raise ValueError("changed/short")
        return raw
    finally: os.close(fd)
def close(a,b,tol=3e-13): return abs(a-b)<=tol*max(1.0,abs(b))

def verify(root):
    c=Checks(); root=root.resolve(strict=True)
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    c.require(actual==FILES,"exact eight-file closure")
    c.require(all(not p.is_symlink() for p in root.rglob("*")),"no links")
    held={name:held_read(root/Path(name)) for name in FILES}
    manifest=strict_json(held[MANIFEST]); rows=manifest.get("entries")
    c.require(manifest.get("schema")=="same_layer_alignment_independent_result_audit_manifest_v0","manifest schema")
    c.require(manifest.get("closed_world") is True,"closed world")
    c.require(isinstance(rows,list) and len(rows)==7,"manifest count")
    c.require([x.get("path") for x in rows]==sorted(FILES-{MANIFEST}),"manifest paths")
    for row in rows:
        name=row["path"]
        c.require(PurePosixPath(name).as_posix()==name and ".." not in PurePosixPath(name).parts,"safe path")
        c.require(row["bytes"]==len(held[name]),"manifest bytes: "+name)
        c.require(row["sha256"]==digest(held[name]) and HEX64.fullmatch(row["sha256"]),"manifest hash: "+name)
    pins={"result":"a1ef6fb136027525b6312635cdcca320f05f51c1340c3875b32192454aac1bb3",
          "manifest":"acececae8d58b19126ce69486ae076ee4a1ded956f42935ea6ce921c0c5cd1c0",
          "design":"1c12104da21b87e5979bab1c758f7143b5fdd1d2b47767fa97fe63b7be030653",
          "runner":"8ad7fba62178c5f8f5a761469c27edf16d56fedf4be97304ef4acd14af703d04"}
    for path,key in (("evidence/runpod_result.json","result"),("evidence/PACKAGE_MANIFEST.json","manifest"),
                     ("evidence/design_lock.json","design"),("evidence/same_layer_alignment_oracle.py","runner")):
        c.require(digest(held[path])==pins[key],"evidence pin: "+key)
    package_manifest=strict_json(held["evidence/PACKAGE_MANIFEST.json"])
    c.require(package_manifest["schema"]=="same-layer-expert-alignment-source-manifest-v0","source manifest schema")
    c.require(package_manifest["files"]["design_lock.json"]==pins["design"] and
              package_manifest["files"]["same_layer_alignment_oracle.py"]==pins["runner"],"sealed source entries")
    design=strict_json(held["evidence/design_lock.json"]); result=strict_json(held["evidence/runpod_result.json"])
    c.require(result["schema"]=="same-layer-expert-alignment-superoracle-result-v0","result schema")
    c.require(result["bindings"]["design_sha256"]==pins["design"] and
              result["bindings"]["script_sha256"]==pins["runner"] and
              result["bindings"]["source_manifest_sha256"]==design["source"]["manifest_sha256"],"binding chain")
    c.require(tuple(design["source"]["experts"])==EXPERTS and tuple(design["source"]["roles"])==ROLES,"frozen identities")
    sources=result["bindings"]["sources"]
    c.require(len(sources)==32,"32 source receipts")
    expected_pairs=[(expert,role) for expert in EXPERTS for role in ROLES]
    c.require([(x["expert"],x["role"]) for x in sources]==expected_pairs,"source receipt order/mapping")
    c.require(len({x["sha256"] for x in sources})==32,"unique source hashes")
    for row in sources:
        expected=f"qwen_weight_cache/rd_structure_diag_cross_expert/l15e{row['expert']}_{row['role']}.bf16.bin"
        c.require(row["bytes"]==3_145_728 and row["local_path"]==expected and HEX64.fullmatch(row["sha256"]),
                  f"source receipt structure: {row['expert']}/{row['role']}")
    scored=result["scored"]
    c.require(len(scored)==32 and [(x["target_expert"],x["role"]) for x in scored]==expected_pairs,"32 scored rows/order")
    c.require(len({x["selection_sha256"] for x in scored})==32 and
              all(HEX64.fullmatch(x["selection_sha256"]) for x in scored),"opaque selection receipts")
    captured=energy=0.0; role_captured={r:0.0 for r in ROLES}; role_energy={r:0.0 for r in ROLES}
    for row in scored:
        c.require(row["source_energy"]>0 and row["captured_energy"]>=0 and
                  close(row["capture"],row["captured_energy"]/row["source_energy"]),"target capture arithmetic")
        captured+=row["captured_energy"]; energy+=row["source_energy"]
        role_captured[row["role"]]+=row["captured_energy"]; role_energy[row["role"]]+=row["source_energy"]
    pooled=captured/energy
    c.require(close(result["capture"],pooled),"pooled capture")
    c.require(close(role_captured["up"]/role_energy["up"],0.012359993622486992) and
              close(role_captured["down"]/role_energy["down"],0.018643001979871822),"role captures")
    missing=design["decision"]["existing_composite_missing_s_bpw"]
    required=1-2**(-2*missing)
    c.require(close(result["required_capture"],required) and
              close(required,design["decision"]["required_up_down_capture_if_sole_missing_module"]),"required capture")
    c.require(result["absolute_capture_cushion"]==design["oracle"]["absolute_capture_cushion"]==.001,"cushion")
    favourable=min(1.0,pooled+.001); shortfall=required-favourable
    c.require(close(result["favourable_capture"],favourable) and close(result["shortfall"],shortfall),"favourable/shortfall")
    c.require(favourable<required and result["status"]=="HARD_KILL_SAME_LAYER_UP_DOWN_ANCESTRY","narrow ancestry kill")
    c.require("Gate and nonlinear/activation-aware codecs are not covered" in result["claim_boundary"],"claim boundary")
    c.require(result["access"]=={"auxiliary_payloads_opened":32,"pinned_payloads_opened":0,
                                  "fresh_validation_payloads_opened":0},"access ledger")
    source=held["evidence/same_layer_alignment_oracle.py"].decode("utf-8")
    c.require("reference_experts = [expert for expert in EXPERTS if expert != target_expert]" in source and
              "reference_experts[index // ROWS]" in source and "selected % ROWS" in source,
              "sealed mapping excludes target and maps index structurally")
    c.require("np.stack((selected_expert, selected_row), axis=1).astype(\"<i8\")" in source,
              "selection receipt serialization structure")
    # Result-only limitation: hashes bind mappings but no selected (expert,row) pairs are emitted.
    c.require(all(set(x)=={"target_expert","role","captured_energy","source_energy","capture","selection_sha256"}
                  for x in scored),"selection mappings are hash-only")
    receipt=strict_json(held["audit_receipt.json"])
    c.require(receipt["schema"]=="same_layer_alignment_independent_result_audit_receipt_v0" and
              receipt["verdict"]=="PASS_NARROW_UP_DOWN_ANCESTRY_KILL_WITH_SELECTION_RECEIPT_LIMITATION","receipt verdict")
    c.require(receipt["selection_receipt_assessment"]["sealed_mapping_construction_structurally_valid"] is True and
              receipt["selection_receipt_assessment"]["mapping_replayable_from_result_only"] is False,"selection limitation")
    c.require(receipt["result_origin_path"]==
              "research/same_layer_expert_alignment_superoracle_v0_runpod_result_20260901/result.json","sibling origin")
    body=dict(receipt); claim=body.pop("receipt_sha256",None)
    c.require(bool(HEX64.fullmatch(str(claim))) and digest(canonical(body))==claim,"receipt seal")
    c.require(receipt["result_sha256"]==pins["result"] and receipt["verifier_check_count"]==c.count+1,"receipt result/count")
    return c.count

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--audit-dir",type=Path,default=Path(__file__).resolve().parent);a=p.parse_args(argv)
    try:n=verify(a.audit_dir)
    except Exception as e:
        print(json.dumps({"schema":"same_layer_alignment_independent_result_verify_v0","verdict":"BLOCK","error":f"{type(e).__name__}: {e}"},sort_keys=True));return 1
    print(json.dumps({"schema":"same_layer_alignment_independent_result_verify_v0","verdict":"PASS_WITH_LIMITATION","checks":n,
                      "manifest_sha256":digest(held_read(a.audit_dir.resolve()/MANIFEST))},sort_keys=True));return 0
if __name__=="__main__":sys.exit(main())
