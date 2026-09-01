#!/usr/bin/env python3
"""Independent, result-only standard-library audit for QSB-PTQ-v1."""

from __future__ import annotations

import argparse, hashlib, json, math, os, re, stat, sys
from fractions import Fraction
from pathlib import Path, PurePosixPath

MANIFEST = "AUDIT_MANIFEST.json"
FILES = {
    "README.md", "audit_receipt.json", "verify_audit.py", MANIFEST,
    "evidence/PACKAGE_MANIFEST.json", "evidence/design_lock.json",
    "evidence/panel_bindings.json", "evidence/runpod_result.json", "evidence/stage0_screen.py",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PANEL_VALUES = 28_311_552
FIT_EXPERTS = (0, 2, 4)
ROLES = ("gate", "up", "down")

class Checks:
    def __init__(self): self.count = 0
    def require(self, condition, label):
        self.count += 1
        if not condition: raise AssertionError(f"check {self.count} failed: {label}")

def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")
def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError("duplicate key: " + key)
            out[key] = value
        return out
    def finite(text):
        value = float(text)
        if not math.isfinite(value): raise ValueError("nonfinite number")
        return value
    def constant(text): raise ValueError("nonfinite token: " + text)
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_float=finite, parse_constant=constant)
def held_read(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink(): raise ValueError("non-regular evidence")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd); parts = []
        if (before.st_dev, before.st_ino, before.st_size) != (opened.st_dev, opened.st_ino, opened.st_size):
            raise ValueError("evidence identity changed")
        while True:
            block = os.read(fd, 1 << 20)
            if not block: break
            parts.append(block)
        after = os.fstat(fd)
        if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("evidence changed")
        raw = b"".join(parts)
        if len(raw) != opened.st_size: raise ValueError("short read")
        return raw
    finally: os.close(fd)
def close(a, b, tolerance=3e-13): return abs(a-b) <= tolerance * max(1.0, abs(b))
def jackknife(expert_sse, expert_energy):
    total_sse, total_energy = math.fsum(expert_sse), math.fsum(expert_energy)
    estimate = 1.0 - total_sse / total_energy
    deletes = [1.0 - (total_sse-expert_sse[i])/(total_energy-expert_energy[i]) for i in range(6)]
    center = math.fsum(deletes)/6
    se = math.sqrt(5/6 * math.fsum((x-center)**2 for x in deletes))
    return estimate, deletes, center, se

def verify(root):
    c = Checks(); root = root.resolve(strict=True)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    c.require(actual == FILES, "exact nine-file closure")
    c.require(all(not p.is_symlink() for p in root.rglob("*")), "no links in closure")
    held = {name: held_read(root / Path(name)) for name in FILES}
    manifest = strict_json(held[MANIFEST])
    c.require(manifest.get("schema") == "qsb_ptq_v1_independent_result_audit_manifest_v0", "manifest schema")
    c.require(manifest.get("closed_world") is True, "closed world")
    entries = manifest.get("entries")
    c.require(isinstance(entries, list) and len(entries) == 8, "manifest entries")
    c.require([x.get("path") for x in entries] == sorted(FILES-{MANIFEST}), "manifest exact paths")
    for row in entries:
        name = row["path"]
        c.require(PurePosixPath(name).as_posix() == name and ".." not in PurePosixPath(name).parts,
                  "safe path: " + name)
        c.require(row["bytes"] == len(held[name]), "bytes: " + name)
        c.require(row["sha256"] == digest(held[name]) and HEX64.fullmatch(row["sha256"]),
                  "hash: " + name)

    expected = {
        "result": "c9b8d26e8d09bbb225288e7787d9ef0154a6a958cc4cf1798cd2d3358b3eeae2",
        "source_manifest": "aaae42924d01f3508bd66ad8efa586549a687e69fe6a8f31bdcc9d1c8806707f",
        "design": "bbf618ae4a6a3b1195e8c844cefb1e47c492b2330094ffa83fc45cc63af847f7",
        "bindings": "5c5f1b6afcb85b91591f44ba8165ae4067a93b53e056eec1e20d8624a081df0d",
        "runner": "8e697f3f1dde5e79cec7619a00366d009a3c926f3de03650358be625bf4cacfe",
    }
    for name, key in (("evidence/runpod_result.json", "result"),
                      ("evidence/PACKAGE_MANIFEST.json", "source_manifest"),
                      ("evidence/design_lock.json", "design"),
                      ("evidence/panel_bindings.json", "bindings"),
                      ("evidence/stage0_screen.py", "runner")):
        c.require(digest(held[name]) == expected[key], "external pin: " + key)
    source_manifest = strict_json(held["evidence/PACKAGE_MANIFEST.json"])
    source_rows = {row["path"]: row for row in source_manifest["entries"]}
    for path, evidence in (("design_lock.json", "evidence/design_lock.json"),
                           ("panel_bindings.json", "evidence/panel_bindings.json"),
                           ("stage0_screen.py", "evidence/stage0_screen.py")):
        c.require(source_rows[path]["bytes"] == len(held[evidence]) and
                  source_rows[path]["sha256"] == digest(held[evidence]), "sealed source entry: " + path)

    result = strict_json(held["evidence/runpod_result.json"])
    design = strict_json(held["evidence/design_lock.json"])
    bindings = strict_json(held["evidence/panel_bindings.json"])
    c.require(result["schema"] == "qwen_stochastic_binary_channel_ptq_stage0_result_v1", "result schema")
    c.require(result["status"] == "POLICY_REJECT_ALL_RATE_CELLS", "overall status")
    c.require(result["bindings"]["design_lock_sha256"] == expected["design"] and
              result["bindings"]["panel_bindings_sha256"] == expected["bindings"] and
              result["bindings"]["stage0_script_sha256"] == expected["runner"], "result source hashes")
    c.require(result["bindings"]["plan_sha256"] == bindings["plan"]["sha256"] ==
              design["panel"]["plan_sha256"], "plan hash chain")
    c.require(result["bindings"]["plan_internal_lock_sha256"] == bindings["plan"]["internal_lock_sha256"] ==
              design["panel"]["plan_internal_lock_sha256"], "plan lock chain")
    c.require(len(result["bindings"]["sources"]) == len(bindings["sources"]) == 18, "18 source receipts")
    for a, b in zip(result["bindings"]["sources"], bindings["sources"]):
        c.require(a == {"bytes": b["bytes"], "matrix_ordinal": b["matrix_ordinal"],
                        "relative_path": b["source_relpath"], "sha256": b["sha256"]},
                  "source binding: " + str(b["matrix_ordinal"]))
    c.require(result["fixed_splits"] == {"fit_experts":[0,2,4], "calibration_experts":[1],
                                          "closed_score_experts":[3,5]}, "fixed splits")
    c.require(result["access"] == {"authenticated_panel_sources_opened":18,
                                    "compressed_outputs_created":0,
                                    "fresh_validation_files_opened":0,
                                    "network_operations":0}, "access ledger")
    c.require("Adapted optimistic source-leaking" in result["claim_boundary"] and
              "not independent held-out evidence" in result["claim_boundary"], "claim boundary")
    c.require(result["execution"]["cuda_visible_devices"] == "0" and
              result["execution"]["device"] == "NVIDIA GeForce RTX 5090" and
              result["execution"]["elapsed_seconds"] > 0, "runtime")

    rate_specs = [("QSB215",309,Fraction(155,72),1_261_568),
                  ("QSB230",331,Fraction(83,36),1_351_680),
                  ("QSB250",359,Fraction(5,2),1_466_368)]
    c.require(len(result["cells"]) == 3, "three cells")
    for row, (name, pages, rate, payload_bytes) in zip(result["cells"], rate_specs):
        cell, oracle = row["cell"], row["qwen_oracle"]
        c.require(cell["cell"] == name and cell["pages"] == pages, "cell identity: "+name)
        c.require(cell["payload_bytes_per_expert"] == payload_bytes and
                  cell["container_bytes"]*8 == PANEL_VALUES*rate.numerator//rate.denominator,
                  "physical byte ledger: "+name)
        c.require(close(cell["physical_bpw"], float(rate)) and close(cell["payload_bpw"]+cell["metadata_bpw"],
                  float(rate)) and cell["cold_read_amplification"] < 2, "rate/read ledger: "+name)
        physical_bits = payload_bytes*8; fit_target = .965*physical_bits; limit = .97*physical_bits
        c.require(oracle["fit_target_fraction"] == .965 and close(oracle["fit_target_bits_by_expert"],fit_target),
                  "fit target: "+name)
        c.require(oracle["execution_limit_fraction"] == .97 and close(oracle["execution_limit_bits_by_expert"],limit)
                  and oracle["physical_reservoir_bits_by_expert"] == physical_bits, "rate limits: "+name)
        kls = oracle["ideal_kl_bits_by_expert"]
        c.require(len(kls)==6 and close(math.fsum(kls[i] for i in FIT_EXPERTS)/3, fit_target),
                  "fit-expert KL mean: "+name)
        for i, value in enumerate(kls):
            c.require(close(oracle["execution_limit_margin_bits_by_expert"][i],limit-value) and
                      close(oracle["physical_reservoir_margin_bits_by_expert"][i],physical_bits-value),
                      f"rate margins: {name}/{i}")
            c.require(value < limit and value < physical_bits, f"no rate kill: {name}/{i}")
        matrices = oracle["matrix_rows"]
        c.require(len(matrices)==18 and [x["matrix_ordinal"] for x in matrices]==list(range(18)),
                  "matrix rows: "+name)
        expert_sse=[0.0]*6; expert_energy=[0.0]*6; role_sse=[0.0]*3; role_energy=[0.0]*3
        for i, matrix in enumerate(matrices):
            expert=i//3; role=i%3
            c.require(matrix["expert_ordinal"]==expert and matrix["role"]==ROLES[role] and
                      matrix["feature_rank"]==257, f"matrix identity/rank: {name}/{i}")
            c.require(close(matrix["capture"],1-matrix["sse"]/matrix["source_energy"]),
                      f"matrix capture: {name}/{i}")
            expert_sse[expert]+=matrix["sse"]; expert_energy[expert]+=matrix["source_energy"]
            role_sse[role]+=matrix["sse"]; role_energy[role]+=matrix["source_energy"]
        c.require(close(oracle["sse"],math.fsum(expert_sse)) and
                  close(oracle["source_energy"],math.fsum(expert_energy)), "pooled sums: "+name)
        pooled_capture=1-oracle["sse"]/oracle["source_energy"]
        c.require(close(oracle["capture"],pooled_capture), "pooled capture: "+name)
        for i,x in enumerate(oracle["experts"]):
            c.require(x["expert_ordinal"]==i and close(x["sse"],expert_sse[i]) and
                      close(x["source_energy"],expert_energy[i]) and
                      close(x["capture"],1-expert_sse[i]/expert_energy[i]), f"expert fold: {name}/{i}")
        for i,x in enumerate(oracle["roles"]):
            c.require(x["role"]==ROLES[i] and close(x["sse"],role_sse[i]) and
                      close(x["source_energy"],role_energy[i]) and
                      close(x["capture"],1-role_sse[i]/role_energy[i]), f"role fold: {name}/{i}")
        estimate,deletes,center,se=jackknife(expert_sse,expert_energy); u=oracle["uncertainty"]
        c.require(close(u["estimate"],estimate) and close(u["jackknife_center"],center) and
                  close(u["jackknife_se"],se), "jackknife summary: "+name)
        c.require(all(close(a,b) for a,b in zip(u["delete_one_expert"],deletes)) and
                  close(u["lower_three_se"],estimate-3*se) and close(u["upper_three_se"],estimate+3*se),
                  "jackknife bounds: "+name)
        required=1-.8*2**(-2*float(rate)); relative_mse=oracle["sse"]/oracle["source_energy"]
        f_value=relative_mse*2**(2*float(rate))
        c.require(close(oracle["required_capture"],required), "required capture: "+name)
        c.require(f_value > .8 and u["upper_three_se"] < required and
                  oracle["status"]=="HARD_KILL_FAVOURABLE_ORACLE_UCB_BELOW_EXACT_REQUIREMENT",
                  "favourable hard kill: "+name)
        c.require(row["matched_gaussian_controls"]==[] and
                  row["control_gate"]=={"status":"NOT_RUN_ORACLE_DID_NOT_SURVIVE"},
                  "no controls after kill: "+name)

    receipt = strict_json(held["audit_receipt.json"])
    c.require(receipt["schema"]=="qsb_ptq_v1_independent_result_audit_receipt_v0" and
              receipt["verdict"]=="PASS_HARD_KILL_NO_CONTROLS", "receipt schema/verdict")
    body=dict(receipt); claimed=body.pop("receipt_sha256",None)
    c.require(bool(HEX64.fullmatch(str(claimed))) and digest(canonical(body))==claimed, "receipt seal")
    c.require(receipt["result_origin_path"]==
              "research/qwen_stochastic_binary_channel_ptq_stage0_v1_runpod_result_20260901/result.json",
              "sibling result origin")
    c.require(receipt["result_sha256"]==expected["result"] and receipt["verifier_check_count"]==c.count+1,
              "receipt result/count")
    return c.count

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--audit-dir",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args(argv)
    try: checks=verify(args.audit_dir)
    except Exception as e:
        print(json.dumps({"schema":"qsb_ptq_v1_independent_result_verify_v0","verdict":"BLOCK","error":f"{type(e).__name__}: {e}"},sort_keys=True)); return 1
    print(json.dumps({"schema":"qsb_ptq_v1_independent_result_verify_v0","verdict":"PASS","checks":checks,
                      "manifest_sha256":digest(held_read(args.audit_dir.resolve()/MANIFEST))},sort_keys=True)); return 0
if __name__=="__main__": sys.exit(main())
