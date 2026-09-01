#!/usr/bin/env python3
"""Result-only standard-library audit of the decoded affine residual oracle."""

from __future__ import annotations
import argparse, hashlib, json, math, os, re, stat, sys
from pathlib import Path, PurePosixPath

MANIFEST="AUDIT_MANIFEST.json"
FILES={"README.md","audit_receipt.json","verify_audit.py",MANIFEST,
       "evidence/README.md","evidence/decoded_affine_oracle.py","evidence/runpod_result.json"}
HEX64=re.compile(r"^[0-9a-f]{64}$")
WIDTHS=(2048,512,128,32); MODES=("scale","bias","affine"); ROLES=("gate","up","down")
class Checks:
    def __init__(self): self.count=0
    def require(self,condition,label):
        self.count+=1
        if not condition: raise AssertionError(f"check {self.count} failed: {label}")
def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
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
    if not stat.S_ISREG(before.st_mode) or path.is_symlink(): raise ValueError("non-regular/link evidence")
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_BINARY",0)|getattr(os,"O_NOFOLLOW",0))
    try:
        opened=os.fstat(fd); chunks=[]
        if (before.st_dev,before.st_ino,before.st_size)!=(opened.st_dev,opened.st_ino,opened.st_size): raise ValueError("identity changed")
        while True:
            x=os.read(fd,1<<20)
            if not x: break
            chunks.append(x)
        after=os.fstat(fd); raw=b"".join(chunks)
        if (opened.st_size,opened.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or len(raw)!=opened.st_size: raise ValueError("changed/short read")
        return raw
    finally: os.close(fd)
def close(a,b,tol=3e-13): return abs(a-b)<=tol*max(1.0,abs(b))

def verify(root):
    c=Checks(); root=root.resolve(strict=True)
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    c.require(actual==FILES,"exact seven-file closure")
    c.require(all(not p.is_symlink() for p in root.rglob("*")),"no links")
    held={name:held_read(root/Path(name)) for name in FILES}
    manifest=strict_json(held[MANIFEST]); entries=manifest.get("entries")
    c.require(manifest.get("schema")=="decoded_affine_residual_independent_result_audit_manifest_v0","manifest schema")
    c.require(manifest.get("closed_world") is True,"closed world")
    c.require(isinstance(entries,list) and len(entries)==6,"manifest count")
    c.require([x.get("path") for x in entries]==sorted(FILES-{MANIFEST}),"manifest paths")
    for row in entries:
        name=row["path"]
        c.require(PurePosixPath(name).as_posix()==name and ".." not in PurePosixPath(name).parts,"safe path: "+name)
        c.require(row["bytes"]==len(held[name]),"manifest bytes: "+name)
        c.require(row["sha256"]==digest(held[name]) and HEX64.fullmatch(row["sha256"]),"manifest hash: "+name)
    pins={"result":"e8cff5c3ee45dc209c4b4d5368e060953e4ccc8e08047a28bddca8e36e19de35",
          "script":"7d06910a9cdb66349bf3900ee0eb7cc27b08883b9f943ac33ab4f2c4e4a11f53",
          "readme":"68a2363393aa7be61b7fcaf91cdab81b407c52998cf0187775ae7b57d9cd0bd4"}
    for path,key in (("evidence/runpod_result.json","result"),("evidence/decoded_affine_oracle.py","script"),("evidence/README.md","readme")):
        c.require(digest(held[path])==pins[key],"evidence pin: "+key)
    # There was no pre-execution package manifest or receipt. These checks authenticate
    # only the current copied script/result bytes, not their temporal ordering.
    c.require("MANIFEST" not in held["evidence/README.md"].decode("utf-8").upper(),"no manifest claimed by README")
    result=strict_json(held["evidence/runpod_result.json"]); source=held["evidence/decoded_affine_oracle.py"].decode("utf-8")
    c.require(result["schema"]=="decoded-affine-residual-oracle-v0","result schema")
    c.require(result["bindings"]["script_sha256"]==pins["script"],"result/current-script hash")
    c.require(all(HEX64.fullmatch(result["bindings"][k]) for k in
                  ("plan_sha256","header_sha256","decoded_post_klt_sha256","script_sha256")),"binding hash syntax")
    c.require("BASELINE_F = 0.9888693569009007" in source and "TARGET_F = 0.8" in source,
              "current script baseline/target constants")
    c.require("favourable_net_f = BASELINE_F * ratio * 2.0 ** (2.0 * side_bpw)" in source,
              "current script transfer formula")
    c.require("exact fp64 source-fitted coefficients" in result["claim_boundary"].lower() and
              "nominal fp16 coefficient bits" in result["claim_boundary"].lower(),"favourable claim boundary")
    matrices=result["matrices"]
    c.require(len(matrices)==18 and [x["matrix_ordinal"] for x in matrices]==list(range(18)),"18 matrix rows")
    baseline_sse=math.fsum(x["baseline_sse"] for x in matrices)
    source_energy=math.fsum(x["source_energy"] for x in matrices)
    baseline=result["baseline"]
    c.require(close(baseline["sse"],baseline_sse) and close(baseline["source_energy"],source_energy),"baseline sums")
    c.require(close(baseline["relative_mse"],baseline_sse/source_energy),"baseline relative MSE")
    c.require(close(baseline["F"],baseline["relative_mse"]*2**(2*2.5)),"baseline 2.5-bpw F")
    c.require(close(baseline["F"],0.9888693569009007) and result["target_F"]==0.8,"baseline/target constants")
    totals={(mode,width):0.0 for mode in MODES for width in WIDTHS}; two_way=0.0
    for i,row in enumerate(matrices):
        c.require(row["role"]==ROLES[i%3] and row["baseline_sse"]>=0 and row["source_energy"]>0,
                  "matrix identity/finite positive")
        c.require(set(row["oracle_sse"])=={f"{mode}_w{width}" for mode in MODES for width in WIDTHS},
                  "matrix oracle family")
        for mode in MODES:
            for width in WIDTHS:
                value=row["oracle_sse"][f"{mode}_w{width}"]
                c.require(0<=value<=row["baseline_sse"]+1e-9,f"favourable matrix SSE: {i}/{mode}/{width}")
                totals[(mode,width)]+=value
        c.require(0<=row["two_way_bias_sse"]<=row["baseline_sse"]+1e-9,"two-way matrix SSE")
        two_way+=row["two_way_bias_sse"]
    cells=result["cells"]
    c.require(len(cells)==13,"13 frozen correction cells")
    by_key={(x["family"],x["width"]):x for x in cells}
    c.require(len(by_key)==13,"unique correction cells")
    for mode in MODES:
        coeffs=1 if mode in ("scale","bias") else 2
        for width in WIDTHS:
            row=by_key[(mode,width)]; sse=totals[(mode,width)]; side=16*coeffs/width; ratio=sse/baseline_sse
            f=baseline["F"]*ratio*2**(2*side)
            c.require(close(row["exact_fp64_oracle_sse"],sse) and close(row["fraction_of_baseline_sse"],ratio),
                      f"cell SSE/ratio: {mode}/{width}")
            c.require(close(row["nominal_fp16_coefficient_bpw"],side) and close(row["favourable_transfer_F"],f),
                      f"cell rate/F: {mode}/{width}")
            c.require(row["passes_target"]==(f<=result["target_F"]),f"cell decision: {mode}/{width}")
    row=by_key[("row_plus_column_bias",None)]; side=16*(768+2048-1)/(768*2048); ratio=two_way/baseline_sse
    f=baseline["F"]*ratio*2**(2*side)
    c.require(close(row["exact_fp64_oracle_sse"],two_way) and close(row["fraction_of_baseline_sse"],ratio),
              "two-way SSE/ratio")
    c.require(close(row["nominal_fp16_coefficient_bpw"],side) and close(row["favourable_transfer_F"],f),
              "two-way rate/F")
    c.require(row["passes_target"]==(f<=result["target_F"]),"two-way decision")
    ordered=sorted(cells,key=lambda x:(x["favourable_transfer_F"],x["family"],-1 if x["width"] is None else -x["width"]))
    c.require(result["best"]==ordered[0] and cells==ordered,"best and frozen ordering")
    c.require(all(not x["passes_target"] and x["favourable_transfer_F"]>0.8 for x in cells),"all favorable cells fail")
    c.require(result["status"]=="HARD_KILL_AFFINE_CORRECTION_FAMILY","favorable-family hard kill")
    c.require(result["runtime"]["elapsed_seconds"]>0 and result["runtime"]["device"]=="b'NVIDIA GeForce RTX 5090'",
              "runtime ledger")
    receipt=strict_json(held["audit_receipt.json"])
    c.require(receipt["schema"]=="decoded_affine_residual_independent_result_audit_receipt_v0" and
              receipt["verdict"]=="PASS_NUMERICAL_KILL_WITH_UNSEALED_PREEXECUTION_PROVENANCE_LIMITATION",
              "receipt verdict")
    c.require(receipt["preexecution_source_manifest_present"] is False and
              receipt["scope"]["preexecution_preregistration_authenticated"] is False,
              "provenance limitation recorded")
    body=dict(receipt); claim=body.pop("receipt_sha256",None)
    c.require(bool(HEX64.fullmatch(str(claim))) and digest(canonical(body))==claim,"receipt seal")
    c.require(receipt["result_origin_path"]==
              "research/decoded_affine_residual_oracle_v0_runpod_result_20260901/result.json",
              "sibling result origin")
    c.require(receipt["result_sha256"]==pins["result"] and receipt["verifier_check_count"]==c.count+1,
              "receipt result/count")
    return c.count

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--audit-dir",type=Path,default=Path(__file__).resolve().parent); a=p.parse_args(argv)
    try: n=verify(a.audit_dir)
    except Exception as e:
        print(json.dumps({"schema":"decoded_affine_residual_independent_result_verify_v0","verdict":"BLOCK","error":f"{type(e).__name__}: {e}"},sort_keys=True)); return 1
    print(json.dumps({"schema":"decoded_affine_residual_independent_result_verify_v0","verdict":"PASS_WITH_LIMITATION","checks":n,
                      "manifest_sha256":digest(held_read(a.audit_dir.resolve()/MANIFEST))},sort_keys=True)); return 0
if __name__=="__main__": sys.exit(main())
