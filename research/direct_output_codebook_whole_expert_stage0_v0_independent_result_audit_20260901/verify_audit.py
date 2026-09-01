#!/usr/bin/env python3
"""Independent standard-library result/binary audit for the direct d8 codebook cell."""

from __future__ import annotations
import argparse, hashlib, json, math, os, re, stat, struct, sys
from pathlib import Path, PurePosixPath

MANIFEST="AUDIT_MANIFEST.json"
SEEDS=(2026090111,2026090112)
LABELS=("source","gaussian")
FILES={"README.md","audit_receipt.json","verify_audit.py",MANIFEST,
       "evidence/SOURCE_MANIFEST.json","evidence/design_lock.json","evidence/direct_output_codebook_stage0.py",
       "evidence/runpod_result/result.json"}
for seed in SEEDS:
    for label in LABELS:
        FILES.add(f"evidence/runpod_result/seed_{seed}/{label}_global_side.bin")
        FILES.add(f"evidence/runpod_result/seed_{seed}/{label}_row_moments.bin")
HEX64=re.compile(r"^[0-9a-f]{64}$")
PANEL_VALUES=28_311_552
GLOBAL_HEADER=4096; CODE_COUNT=32768; DIM=8; INDEX_BITS=15
CODEBOOK_BYTES=CODE_COUNT*DIM*2; GLOBAL_BYTES=GLOBAL_HEADER+CODEBOOK_BYTES
MOMENT_BYTES=18*768*2*2

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
def all_fp16_finite(raw):
    return len(raw)%2==0 and all((word&0x7c00)!=0x7c00 for (word,) in struct.iter_unpack("<H",raw))
def round_up(value,unit): return ((value+unit-1)//unit)*unit

def verify(root):
    c=Checks(); root=root.resolve(strict=True)
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    c.require(actual==FILES,"exact sixteen-file closure")
    c.require(all(not p.is_symlink() for p in root.rglob("*")),"no links")
    held={name:held_read(root/Path(name)) for name in FILES}
    manifest=strict_json(held[MANIFEST]); entries=manifest.get("entries")
    c.require(manifest.get("schema")=="direct_output_d8_independent_result_audit_manifest_v0","manifest schema")
    c.require(manifest.get("closed_world") is True,"closed world")
    c.require(isinstance(entries,list) and len(entries)==15,"manifest count")
    c.require([x.get("path") for x in entries]==sorted(FILES-{MANIFEST}),"manifest paths")
    for row in entries:
        name=row["path"]
        c.require(PurePosixPath(name).as_posix()==name and ".." not in PurePosixPath(name).parts,"safe path: "+name)
        c.require(row["bytes"]==len(held[name]),"manifest bytes: "+name)
        c.require(row["sha256"]==digest(held[name]) and HEX64.fullmatch(row["sha256"]),"manifest hash: "+name)
    pins={"result":"701e3ebe3da89c4fa5e72bca335a1b33e7bb57937851956ee139630d26b9b991",
          "design":"397cf9e68cdf661b0ac6dddddfc1199275e792848bbd69e10af935ae9758e135",
          "runner":"c985ebf19343dc275e0d49ceedabba387fc67e991dac38f0da7bbbc8d7bfc7c4",
          "source_manifest":"1486475e08c4dd92beda97d60ea6290b468ea9b88beb9bc9b21a460d5ff0406a"}
    for path,key in (("evidence/runpod_result/result.json","result"),("evidence/design_lock.json","design"),
                     ("evidence/direct_output_codebook_stage0.py","runner"),("evidence/SOURCE_MANIFEST.json","source_manifest")):
        c.require(digest(held[path])==pins[key],"evidence pin: "+key)
    source_manifest=strict_json(held["evidence/SOURCE_MANIFEST.json"])
    c.require(source_manifest["schema"]=="direct-output-codebook-stage0-source-manifest-v0","source manifest schema")
    source_entries={x["path"]:x for x in source_manifest["files"]}
    for path,evidence in (("design_lock.json","evidence/design_lock.json"),
                          ("direct_output_codebook_stage0.py","evidence/direct_output_codebook_stage0.py")):
        c.require(source_entries[path]["bytes"]==len(held[evidence]) and source_entries[path]["sha256"]==digest(held[evidence]),
                  "sealed source entry: "+path)
    result=strict_json(held["evidence/runpod_result/result.json"]); design=strict_json(held["evidence/design_lock.json"])
    c.require(result["schema"]=="direct-output-codebook-whole-expert-stage0-result-v0","result schema")
    c.require(result["status"]=="KILL" and result["decision_reasons"]==
              ["both fixed direct-output seeds fail the favorable held-out oracle"],"kill decision")
    c.require(design["architecture"]["vector_dimension"]==8 and design["architecture"]["code_count"]==CODE_COUNT and
              design["architecture"]["index_bits_per_vector"]==INDEX_BITS,"narrow direct d8 cell")
    c.require("not a converse for arbitrary VQ" in result["claim_boundary"],"narrow claim boundary")
    lock=dict(result); claimed=lock.pop("result_lock_sha256")
    c.require(claimed==digest(canonical(lock)),"canonical result lock")
    c.require(result["source_lock"]["file_sha256"]==design["panel"]["source_lock_file_sha256"] and
              result["source_lock"]["internal_sha256"]==design["panel"]["source_lock_internal_sha256"] and
              result["source_lock"]["bytes"]==design["panel"]["source_lock_bytes"],"source lock binding")
    c.require(result["split"]=={"fit_slots":[0,2,3,5],"holdout_slots":[1,4]},"whole-expert split")
    receipts=result["source_receipts"]
    c.require(len(receipts)==18 and [x["matrix_ordinal"] for x in receipts]==list(range(18)),"source receipts")
    c.require(all(x["declared_sha256"]==x["observed_sha256"] and x["payload_bytes"]==3_145_728 for x in receipts),
              "authenticated receipt hashes/lengths")

    prefix_bytes=GLOBAL_BYTES+6*(1_105_920+9_216+64)
    prefix_bpw=prefix_bytes*8/PANEL_VALUES
    c.require(close(result["six_expert_fixed_prefix_bpw"],prefix_bpw) and close(prefix_bpw,2.0400390625),"six-expert prefix")
    required=.8*2**(-2*prefix_bpw)
    c.require(close(result["six_expert_required_first_stage_relative_residual_energy"],required),"required first-stage q")
    c.require(len(result["six_expert_rate_ledger"])==3,"six-expert ledgers")
    for ledger,requested in zip(result["six_expert_rate_ledger"],(2.15,2.30,2.50)):
        physical=math.ceil(requested*PANEL_VALUES/8); residual=physical-prefix_bytes
        small=residual//6; large=residual%6; local_min=1_115_200+small; local_max=local_min+(1 if large else 0)
        cold=GLOBAL_BYTES+round_up(local_max,4096)
        c.require(ledger["physical_bytes"]==physical and ledger["total_residual_bytes"]==residual,"rate bytes")
        c.require(ledger["local_frame_min_bytes"]==local_min and ledger["local_frame_max_bytes"]==local_max and
                  ledger["large_frame_count"]==large,"frame distribution")
        c.require(close(ledger["actual_bpw"],physical*8/PANEL_VALUES) and
                  close(ledger["residual_bpw"],residual*8/PANEL_VALUES),"bpw arithmetic")
        c.require(ledger["cold_expert_bytes_4k"]==cold and close(ledger["cold_read_amplification"],cold*6/physical),
                  "cold read arithmetic")
    hyp=result["hypothetical_128_expert_layer_ledger"]
    values128=128*4_718_592; prefix128=GLOBAL_BYTES+128*1_115_200
    c.require(close(hyp["fixed_prefix_bpw"],prefix128*8/values128) and
              close(hyp["required_first_stage_relative_residual_energy"],.8*2**(-2*hyp["fixed_prefix_bpw"])),
              "hypothetical 128 prefix/requirement")
    c.require("Arithmetic projection only" in hyp["claim_boundary"],"128 claim boundary")
    for ledger,requested in zip(hyp["rates"],(2.15,2.30,2.50)):
        physical=math.ceil(requested*values128/8); residual=physical-prefix128; small=residual//128; large=residual%128
        local_min=1_115_200+small; local_max=local_min+(1 if large else 0); cold=GLOBAL_BYTES+round_up(local_max,4096)
        c.require(ledger["physical_bytes"]==physical and ledger["total_residual_bytes"]==residual,"128 bytes")
        c.require(ledger["local_frame_min_bytes"]==local_min and ledger["local_frame_max_bytes"]==local_max and
                  ledger["large_frame_count"]==large,"128 frames")
        c.require(close(ledger["cold_read_amplification"],cold*128/physical),"128 cold read")

    moment_reference=None
    artifact_count=0
    for seed_report,seed in zip(result["seed_reports"],SEEDS):
        c.require(seed_report["seed"]==seed,"seed identity")
        for label in LABELS:
            report=seed_report[label]; prefix=f"evidence/runpod_result/seed_{seed}/"
            global_name=prefix+f"{label}_global_side.bin"; moment_name=prefix+f"{label}_row_moments.bin"
            global_raw=held[global_name]; moment_raw=held[moment_name]; artifact_count+=2
            c.require(len(global_raw)==report["global_side_bytes"]==GLOBAL_BYTES and digest(global_raw)==report["global_side_sha256"],
                      f"global artifact: {seed}/{label}")
            c.require(len(moment_raw)==report["row_moments_bytes"]==MOMENT_BYTES and digest(moment_raw)==report["row_moments_sha256"],
                      f"moment artifact: {seed}/{label}")
            fields=struct.unpack_from("<8sIIII",global_raw,0)
            c.require(fields==(b"DOCBS0\0\0",0,CODE_COUNT,DIM,INDEX_BITS),f"global header: {seed}/{label}")
            metadata_raw=global_raw[24:GLOBAL_HEADER].rstrip(b"\0"); metadata=strict_json(metadata_raw)
            c.require(metadata=={"code_count":CODE_COUNT,"control":label=="gaussian","format":"DOCB-WE-S0-v0",
                                 "index_bits":INDEX_BITS,"seed":seed,"vector_dimension":DIM},f"metadata: {seed}/{label}")
            c.require(global_raw[24+len(metadata_raw):GLOBAL_HEADER]==bytes(GLOBAL_HEADER-24-len(metadata_raw)),
                      f"zero header padding: {seed}/{label}")
            c.require(all_fp16_finite(global_raw[GLOBAL_HEADER:]) and all_fp16_finite(moment_raw),
                      f"finite FP16 artifacts: {seed}/{label}")
            if moment_reference is None: moment_reference=moment_raw
            c.require(moment_raw==moment_reference,f"identical row moments: {seed}/{label}")
            evaluation=report["evaluation"]; experts=evaluation["heldout_experts"]
            c.require(len(experts)==2 and [x["slot"] for x in experts]==[1,4],f"heldout experts: {seed}/{label}")
            total_sse=total_energy=0.0
            for expert in experts:
                es=ee=0.0
                c.require(len(expert["matrices"])==3,"three matrices per heldout expert")
                for matrix in expert["matrices"]:
                    c.require(close(matrix["relative_residual_energy"],matrix["sse"]/matrix["source_energy"]),
                              f"matrix q: {seed}/{label}/{matrix['matrix_ordinal']}")
                    c.require(0<matrix["codes_used"]<=CODE_COUNT,"codes used bounds")
                    es+=matrix["sse"]; ee+=matrix["source_energy"]
                c.require(close(expert["sse"],es) and close(expert["source_energy"],ee) and
                          close(expert["relative_residual_energy"],es/ee),f"expert q: {seed}/{label}/{expert['slot']}")
                total_sse+=es; total_energy+=ee
            c.require(close(evaluation["sse"],total_sse) and close(evaluation["source_energy"],total_energy) and
                      close(evaluation["relative_residual_energy"],total_sse/total_energy),f"pooled q: {seed}/{label}")
            factor=2**(2*prefix_bpw); oracle=report["oracle"]
            c.require(close(oracle["relative_residual_energy"],evaluation["relative_residual_energy"]) and
                      close(oracle["F_oracle"],evaluation["relative_residual_energy"]*factor) and
                      close(oracle["s_oracle"],-.5*math.log2(oracle["F_oracle"])),f"pooled F/s: {seed}/{label}")
            for ev,oo in zip(experts,oracle["heldout_experts"]):
                c.require(close(oo["F_oracle"],ev["relative_residual_energy"]*factor) and
                          close(oo["s_oracle"],-.5*math.log2(oo["F_oracle"])),f"expert F/s: {seed}/{label}")
            checkpoints=[128,256,512,1024,"full_lloyd_1"]
            limits=[(.30,2048),(.25,4096),(.20,8192),(.16,12288),(.14,16000)]
            c.require(report["collapse_reasons"]==[] and len(report["training_trace"])==5,f"no collapse: {seed}/{label}")
            for trace,checkpoint,(maxq,mincodes) in zip(report["training_trace"],checkpoints,limits):
                c.require(trace["checkpoint"]==checkpoint and close(trace["maximum_q"],maxq) and
                          trace["minimum_codes_used"]==mincodes,"checkpoint contract")
                c.require(close(trace["relative_residual_energy"],trace["sse"]/trace["source_energy"]) and
                          trace["relative_residual_energy"]<=maxq and trace["codes_used"]>=mincodes and
                          trace["passed"] is True,"checkpoint passes")
        source_s=seed_report["source"]["oracle"]["s_oracle"]; gaussian_s=seed_report["gaussian"]["oracle"]["s_oracle"]
        c.require(close(seed_report["matched_advantage_s"],source_s-gaussian_s),"matched advantage")
    c.require(artifact_count==8,"all eight artifacts audited")
    c.require(all(x["source"]["oracle"]["F_oracle"]>.8 and x["source"]["collapse_reasons"]==[]
                  for x in result["seed_reports"]),"direct d8 kill without collapse")
    c.require(all(x["matched_advantage_s"]<0 for x in result["seed_reports"]),"both matched advantages negative")
    c.require(len(result["source_moments"])==18,"source moment rows")
    for i,row in enumerate(result["source_moments"]):
        piece=moment_reference[i*3072:(i+1)*3072]
        c.require(row["matrix_ordinal"]==i and row["stored_fp16_bytes"]==3072 and
                  row["stored_fp16_sha256"]==digest(piece),"moment slice binding")
    c.require(len(result["gaussian_moment_match"])==18 and all(x["maximum_row_absolute_mean_error"]<=x["tolerance"] and
              x["maximum_row_absolute_rms_error"]<=x["tolerance"] for x in result["gaussian_moment_match"]),
              "reported Gaussian moment tolerances")
    receipt=strict_json(held["audit_receipt.json"])
    c.require(receipt["schema"]=="direct_output_d8_independent_result_audit_receipt_v0" and
              receipt["verdict"]=="PASS_NARROW_DIRECT_D8_KILL_WITHOUT_COLLAPSE","receipt verdict")
    body=dict(receipt); claim=body.pop("receipt_sha256",None)
    c.require(bool(HEX64.fullmatch(str(claim))) and digest(canonical(body))==claim,"receipt seal")
    c.require(receipt["result_origin_path"]==
              "research/direct_output_codebook_whole_expert_stage0_v0_runpod_result_20260901",
              "sibling result origin")
    c.require(receipt["result_sha256"]==pins["result"] and receipt["verifier_check_count"]==c.count+1,"receipt result/count")
    return c.count

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--audit-dir",type=Path,default=Path(__file__).resolve().parent); a=p.parse_args(argv)
    try: n=verify(a.audit_dir)
    except Exception as e:
        print(json.dumps({"schema":"direct_output_d8_independent_result_verify_v0","verdict":"BLOCK","error":f"{type(e).__name__}: {e}"},sort_keys=True)); return 1
    print(json.dumps({"schema":"direct_output_d8_independent_result_verify_v0","verdict":"PASS","checks":n,
                      "manifest_sha256":digest(held_read(a.audit_dir.resolve()/MANIFEST))},sort_keys=True)); return 0
if __name__=="__main__": sys.exit(main())
