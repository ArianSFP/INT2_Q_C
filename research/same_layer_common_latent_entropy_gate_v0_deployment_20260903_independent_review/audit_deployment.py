"""Payload-blind independent review of one common-latent deployment copy."""
from __future__ import annotations
import argparse, ast, contextlib, hashlib, importlib.util, io, json, os, stat, sys, tempfile
from pathlib import Path

DEPLOY_MANIFEST="a969382640ad69ee71b6029d901d7eade7b88112d582059d83b947e33d1767c3"
DEPLOY_ROOT="edea8361c0c6d990b9875e0e016e5d31c9cfe525d8803ce2f4d406a2077adae6"
SOURCE_MANIFEST="b92d4b5f307ba1d2b6bc6370d0b7cd118c4ab138dc6c8943402efe632a2a5d8f"
SOURCE_ROOT="f9fe8b64b31edc7599e8e9c302b7e283b2aed9cc24c165916ae3447a9f78311c"
SOURCE_R2_REVIEW_MANIFEST="9273cbc0c503bf6703bfb71e3d5ef6c390690cdc98ebe2a2ae0ef3be9df6ec00"
def req(x,m):
    if not x: raise RuntimeError(m)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def auth(root, expected_manifest, expected_root, schema):
    req(root.is_absolute() and root.is_dir() and not root.is_symlink(),"root")
    mp=root/"SOURCE_MANIFEST.json"; req(sha(mp)==expected_manifest,"manifest hash")
    raw=mp.read_bytes(); obj=json.loads(raw)
    req(raw==(json.dumps(obj,sort_keys=True,separators=(",",":"))+"\n").encode(),"canonical manifest")
    req(obj["schema"]==schema,"schema")
    rows=obj["files"]; names=[x["name"] for x in rows]
    req(names==sorted(names) and len(names)==len(set(names)),"row order")
    req(sorted(p.name for p in root.iterdir())==sorted(names+["SOURCE_MANIFEST.json"]),"closure")
    canon=[]
    for row in rows:
        p=root/row["name"]
        req(set(row)=={"bytes","name","sha256"} and stat.S_ISREG(p.lstat().st_mode) and not p.is_symlink(),"row")
        req(p.stat().st_size==row["bytes"] and sha(p)==row["sha256"],"member pin")
        canon.append({"bytes":row["bytes"],"name":row["name"],"sha256":row["sha256"]})
    root_hash=hashlib.sha256(json.dumps(canon,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    req(root_hash==expected_root==obj["source_root_sha256"],"root hash")
    return obj
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); req(spec and spec.loader,"spec")
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module
def assignments(path):
    result={}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try: result[node.targets[0].id]=ast.literal_eval(node.value)
            except (ValueError,TypeError): pass
    return result
def main():
    p=argparse.ArgumentParser(); p.add_argument("--deployment",required=True); p.add_argument("--source",required=True); p.add_argument("--source-r2-review",required=True); a=p.parse_args()
    dep=Path(a.deployment).resolve(strict=True); src=Path(a.source).resolve(strict=True); sr=Path(a.source_r2_review).resolve(strict=True)
    dm=auth(dep,DEPLOY_MANIFEST,DEPLOY_ROOT,"same_layer_common_latent_deployment_manifest_v0")
    sm=auth(src,SOURCE_MANIFEST,SOURCE_ROOT,"same_layer_common_latent_source_manifest_v0")
    req(dm["parent_source_manifest_sha256"]==SOURCE_MANIFEST,"parent pin")
    req(dm["activation"]=="exactly one AST literal PAYLOAD_EXECUTION_ENABLED changes False to True","activation declaration")
    req(sha(sr/"SOURCE_MANIFEST.json")==SOURCE_R2_REVIEW_MANIFEST,"source review pin")
    sr_receipt=json.loads((sr/"AUDIT_RECEIPT.json").read_bytes())
    req(sr_receipt["status"]=="PASS_REPAIRED_SOURCE_ELIGIBLE_FOR_SEPARATE_DEPLOYMENT_REVIEW" and sr_receipt["source_manifest_sha256"]==SOURCE_MANIFEST,"source approval")
    drows={x["name"]:x for x in dm["files"]}; srows={x["name"]:x for x in sm["files"]}
    req(set(drows)==set(srows),"member names")
    differences=[]
    for name in sorted(drows):
        same=(dep/name).read_bytes()==(src/name).read_bytes()
        if not same: differences.append(name)
    req(differences==["run_gate.py"],"sole differing member")
    sb=(src/"run_gate.py").read_bytes(); db=(dep/"run_gate.py").read_bytes()
    old=b"PAYLOAD_EXECUTION_ENABLED = False"; new=b"PAYLOAD_EXECUTION_ENABLED = True"
    req(sb.count(old)==1 and db.count(new)==1 and sb.replace(old,new)==db,"sole byte transform")
    sa=assignments(src/"run_gate.py"); da=assignments(dep/"run_gate.py")
    changed={k:(sa.get(k),da.get(k)) for k in set(sa)|set(da) if sa.get(k)!=da.get(k)}
    req(changed=={"PAYLOAD_EXECUTION_ENABLED":(False,True)},"sole literal change")
    req(da["PANEL_LOCK_SHA256"]==sha(dep/"panel_lock.json"),"panel pin")
    req(da["CORE_SHA256"]==sha(dep/"common_latent_core.py"),"core pin")
    req(da["WORKER_SHA256"]==sha(dep/"cupy_worker.py"),"worker pin")
    text=(dep/"run_gate.py").read_text(encoding="utf-8")
    main_at=text.index("def main(")
    order=[text.index(x,main_at) for x in ("if args.authorization != AUTHORIZATION_PHRASE:","if os.environ.get(\"CUDA_VISIBLE_DEVICES\") != \"0\":","package = Path(__file__)","payload_root = Path(args.payload_root)","if not payload_root.is_absolute() or not output.is_absolute():","if not args.payload_root or not payload_root.is_dir() or payload_root.is_symlink():","if not args.output or output.suffix.lower() != \".json\" or output.exists():","if not output.parent.is_dir() or output.parent.is_symlink():","_load_verified_module(","result = worker.run_authorized_panel",'with output.open("x"')]
    req(order==sorted(order),"validation/access order")
    gate=load("common_latent_deployment_review_gate",dep/"run_gate.py")
    class NoPath:
        def __init__(self,*x,**y): raise AssertionError("Path touched before authorization")
    real_path=gate.Path; gate.Path=NoPath
    out=io.StringIO()
    with contextlib.redirect_stdout(out): code=gate.main(["--authorization","WRONG","--payload-root","FORBIDDEN","--output","FORBIDDEN"])
    req(code==2 and json.loads(out.getvalue())["status"]=="HOLD_NO_PAYLOAD_ACCESS","wrong auth")
    gate.Path=real_path
    old_cuda=os.environ.get("CUDA_VISIBLE_DEVICES"); os.environ["CUDA_VISIBLE_DEVICES"]="0"
    try:
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); absent_payload=base/"absent_payload"; absent_output=base/"absent.json"
            try: gate.main(["--authorization",gate.AUTHORIZATION_PHRASE,"--payload-root",str(absent_payload),"--output",str(absent_output)])
            except RuntimeError as e: req(str(e)=="payload root must be an explicit real directory","missing payload rejection")
            else: raise RuntimeError("missing payload accepted")
            empty_payload=base/"empty_payload"; empty_payload.mkdir(); existing=base/"existing.json"; existing.write_text("occupied",encoding="utf-8")
            try: gate.main(["--authorization",gate.AUTHORIZATION_PHRASE,"--payload-root",str(empty_payload),"--output",str(existing)])
            except RuntimeError as e: req(str(e)=="output must be an explicit absent .json file","existing output rejection")
            else: raise RuntimeError("existing output accepted")
            req(existing.read_text(encoding="utf-8")=="occupied","output overwritten")
    finally:
        if old_cuda is None: os.environ.pop("CUDA_VISIBLE_DEVICES",None)
        else: os.environ["CUDA_VISIBLE_DEVICES"]=old_cuda
    req("same_layer_common_latent_cupy_worker_v0" not in sys.modules,"worker imported during negative checks")
    receipt={"schema":"same_layer_common_latent_deployment_independent_review_receipt_v0","status":"PASS_AUTHORIZE_EXACTLY_ONE_QWEN_RUN","auditor_id":"common_latent_source_audit_deployment_review","deployment_manifest_sha256":DEPLOY_MANIFEST,"deployment_root_sha256":DEPLOY_ROOT,"parent_source_manifest_sha256":SOURCE_MANIFEST,"parent_source_root_sha256":SOURCE_ROOT,"parent_source_r2_review_manifest_sha256":SOURCE_R2_REVIEW_MANIFEST,"payload_accessed":False,"payload_member_files_opened":0,"deployment_files_modified":False,"closure":{"member_count_excluding_manifest":len(dm["files"]),"sole_differing_member":"run_gate.py","sole_ast_literal_delta":{"name":"PAYLOAD_EXECUTION_ENABLED","source":False,"deployment":True},"all_other_member_bytes_equal":True},"internal_pins":{"panel":da["PANEL_LOCK_SHA256"],"core":da["CORE_SHA256"],"worker":da["WORKER_SHA256"],"all_exact":True},"guards":{"wrong_authorization_holds_before_Path":True,"CUDA_VISIBLE_DEVICES_exactly_zero_required_before_package_or_payload_path":True,"payload_root_absolute_real_nonsymlink_required_before_worker":True,"output_absolute_absent_json_existing_real_parent_required_before_worker":True,"output_created_with_exclusive_mode_x":True,"existing_output_preserved":True},"authorization":{"authorized_use_count":1,"scope":"one authenticated Qwen layer-15 16-expert Up/Down.T aperture run using this exact deployment manifest","conditions":["CUDA_VISIBLE_DEVICES=0","exact authorization phrase","explicit pinned payload root","explicit absent JSON output under existing real parent","retire this deployment authority after first invocation regardless of scientific outcome"],"not_claimed":"source code does not implement a persistent invocation counter; one-use limit is this signed review authority"},"claim_boundary":"DEPLOYMENT_SOURCE_AUTHORITY_ONLY_NOT_PAYLOAD_RESULT_OR_FINITE_CODEC_EVIDENCE"}
    print(json.dumps(receipt,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
