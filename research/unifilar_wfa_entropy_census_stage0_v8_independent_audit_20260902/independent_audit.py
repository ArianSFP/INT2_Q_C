#!/usr/bin/env python3
"""Independent byte/receipt authenticator for stopped UWFA-SC v8."""

import argparse, hashlib, json, os, re, stat
from pathlib import Path

DOMAIN = b"UWFA-SC-V8-INDEPENDENT-INVENTORY-v1\0"
ROOT = "586b127aa67c88608816a39fbe35888e71d7c00b2928de550420b5e8ae392f18"
EXPECTED = {
 "INDEPENDENT_BOOTSTRAP_ABI.md":(11025,"b46b2703121d2e50460025bc0c5ff53ca28fffb94a1a2b23e58a52ce41bd2160"),
 "README.md":(16067,"ba6d7aa17494dd8a1ef34cf28fa0aea5e3b187443ae6dd3278f61eda3b3c43f2"),
 "container_codec.py":(93379,"645debb547a76818a880bfc346a2dd6230af97b07dc832afb3548a83d6920fed"),
 "cupy_backend.py":(40964,"7904a5e122686487d89fb684b70052507089bfe3bbfe4f1f02520df6ce3fb1ba"),
 "design_lock.json":(11558,"63c5572db0bafebd0ca2129a1f1071a3a0c0d21962168692eb0a2de12323d639"),
 "dispatcher_contract.py":(9205,"747db5747b75074c1191e17055d615df3cddc54da00e29ba03edfd99ddb2a243"),
 "fixture_long_memory.py":(4307,"d72e7c109920f7d2c6a64bcbf9de0c6463ae80b40cbdb3e772af44c30b3a8c38"),
 "fixture_portability.py":(16350,"b8e9c8d0741f5c7de44ad9ae2bedf8ea6b0fba3ec6fa58df80d8d08fb5a8a1db"),
 "protocol.py":(21051,"9e18675a1e646eb10c0900aa3767bff96666943309dbd8db3953c745888d2cc1"),
 "result_envelope.py":(19002,"ad568758b318a9a6f298da2dc17edcd7f7639e2f772511ae680798f301bc4601"),
 "run_source_free_gpu_dev.py":(8263,"888c5420353951d164a76015e6563154df119f1481da29621154a01347791838"),
 "stage0_census.py":(123776,"7b7c2e0fcb6593805e6b2c8234ae59cb42d90fbb7dcf945a35aa5dfe331ae618"),
 "strata_sc_adapter.py":(36184,"08fc8808ac168f6930ee9482e160f25f2bd087829fca4630553aea3510d722c6"),
 "test_source_only.py":(135687,"5dc3730b629dc3c05a1353d036c6a9049013b6c163540c31f2cb8275d5a68383"),
 "universal_adapter.py":(11577,"a5ab2e1919af98c2aa9b3032faa0ba5552efe05cca250bd6844fd48c76aabbc8"),
 "uwfa_common.py":(58875,"db53567ab6d71d5150cc92ef4a78fa9ce5cca01f5474fa2ca32edc8711cc4325"),
 "verify_source.py":(15907,"c9ccbcd0b68681400dab97636bad7e4d445a83f2446d032b53863a8ab77b7714")}

def need(x,m):
 if not x: raise RuntimeError(m)
def sha(x): return hashlib.sha256(x).hexdigest()
def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def strict(x):
 def pairs(rows):
  d={}
  for k,v in rows: need(k not in d,"duplicate JSON key"); d[k]=v
  return d
 return json.loads(x.decode("utf-8"),object_pairs_hook=pairs)
def seal(d,k):
 c=dict(d); v=c.pop(k); need(isinstance(v,str) and re.fullmatch(r"[0-9a-f]{64}",v),k+" syntax"); need(sha(canon(c))==v,k+" mismatch")
def ident(s): return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def readat(fd,name,cap=1<<30):
 need("/" not in name and name not in ("",".",".."),"unsafe name")
 h=os.open(name,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_CLOEXEC",0),dir_fd=fd)
 try:
  a=os.fstat(h); need(stat.S_ISREG(a.st_mode) and 0<=a.st_size<=cap,"member type/size")
  left=a.st_size; out=[]
  while left:
   b=os.read(h,min(1<<20,left)); need(b,"short read"); out.append(b); left-=len(b)
  need(os.read(h,1)==b"","growing member"); z=os.fstat(h); need(ident(a)==ident(z),"changing member")
  return b"".join(out),a
 finally: os.close(h)

def source(path):
 p=path.resolve(strict=True); cur=Path(p.anchor)
 for part in p.parts[1:]: cur/=part; need(not stat.S_ISLNK(os.lstat(cur).st_mode),"symlink ancestor")
 fd=os.open(p,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0))
 try:
  need({e.name for e in os.scandir(fd)}==set(EXPECTED),"source members")
  pre=bytearray(DOMAIN); rows=[]
  for n in sorted(EXPECTED,key=lambda q:q.encode()):
   b,s=readat(fd,n); size,digest=EXPECTED[n]; need(len(b)==s.st_size==size and sha(b)==digest,"source "+n)
   rows.append({"name":n,"bytes":size,"sha256":digest}); pre.extend(n.encode()+b"\0"+str(size).encode()+b"\0"+digest.encode()+b"\n")
  need(sha(pre)==ROOT,"source root"); return rows
 finally: os.close(fd)

def receipt(parent,final,inventory):
 pfd=os.open(parent.resolve(strict=True),os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0)); dfd=-1
 try:
  pn=".uwfa-publish-v8-"+sha(b"UWFA-V8-COMMIT-NAME\0"+final.encode())+".json"
  mb,ms=readat(pfd,pn,1<<20); m=strict(mb); seal(m,"parent_commit_sha256")
  need(m["schema"]=="unifilar-wfa-parent-commit-v8" and m["status"]=="PARENT_MARKER_COMMITTED","marker schema")
  ps=os.fstat(pfd); need((m["parent_device"],m["parent_inode"])==(ps.st_dev,ps.st_ino),"parent inode")
  need((m["commit_marker_device"],m["commit_marker_inode"])==(ms.st_dev,ms.st_ino) and ms.st_nlink==1,"marker identity/link")
  dfd=os.open(final,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0),dir_fd=pfd); ds=os.fstat(dfd)
  need((m["final_directory_device"],m["final_directory_inode"])==(ds.st_dev,ds.st_ino),"directory inode")
  members=m["members"]; need(members==sorted(members,key=lambda r:r["name"].encode()),"member order")
  need(sorted(e.name for e in os.scandir(dfd))==sorted(r["name"] for r in members),"result members")
  blobs={}
  for r in members:
   b,s=readat(dfd,r["name"]); need(len(b)==s.st_size==r["bytes"] and sha(b)==r["sha256"],"result "+r["name"]); blobs[r["name"]]=b
  root=sha(b"UWFA-V8-HELD-DIRECTORY-ROOT\0"+canon({"source_manifest_sha256":m["source_manifest_sha256"],"members":members}))
  need(root==m["directory_root_sha256"],"directory root")
  c=strict(blobs["COMPLETE.json"]); seal(c,"completion_sha256"); need(c["completion_sha256"]==m["completion_sha256"],"completion binding")
  r=strict(blobs["GPU_DEV_RECEIPT.json"]); need(r["status"]=="PASS_SOURCE_FREE_DEVELOPMENT_REPLAY_NO_CLAIM_AUTHORITY","receipt status")
  need(r["payload_authority_granted"] is False and r["public_commit_evidence"] is False,"nonclaim")
  need(r["development_source_inventory"]==inventory and sha(canon(inventory))==r["development_source_root_sha256"]==m["source_manifest_sha256"],"source binding")
  need([x["selector_ordinal"] for x in r["all150"]["cells"]]==list(range(150)),"selectors")
  g=r["independent_gpu_identity"]; need(re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",g["device_uuid"]),"uuid"); need(re.fullmatch(r"[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",g["pci_bus_id"]),"pci")
  return {"receipt_sha256":sha(blobs["GPU_DEV_RECEIPT.json"]),"receipt_bytes":len(blobs["GPU_DEV_RECEIPT.json"]),"parent_marker_sha256":sha(mb),"parent_commit_sha256":m["parent_commit_sha256"],"directory_root_sha256":root,"completion_sha256":c["completion_sha256"],"development_source_root_sha256":r["development_source_root_sha256"],"bound_preflight_sha256":r["bound_source_preflight_receipt_sha256"],"final_device":ds.st_dev,"final_inode":ds.st_ino,"marker_device":ms.st_dev,"marker_inode":ms.st_ino,"marker_links":ms.st_nlink,"gpu_name":g["device_name"],"gpu_uuid":g["device_uuid"],"pci":g["pci_bus_id"]}
 finally:
  if dfd>=0: os.close(dfd)
  os.close(pfd)

if __name__=="__main__":
 a=argparse.ArgumentParser(); a.add_argument("--package",required=True); a.add_argument("--gpu-output-parent",required=True); a.add_argument("--gpu-final-name",required=True); x=a.parse_args()
 inv=source(Path(x.package)); out=receipt(Path(x.gpu_output_parent),x.gpu_final_name,inv)
 print(json.dumps({"status":"PASS_INDEPENDENT_SOURCE_AND_GPU_RECEIPT_AUTHENTICATION","schema":"uwfa-sc-v8-independent-source-only-audit-v1","inventory_root_sha256":ROOT,"inventory_bytes":sum(v[0] for v in EXPECTED.values()),"gpu_receipt":out,"qwen_opened":False,"payload_opened":False},sort_keys=True,separators=(",",":")))
