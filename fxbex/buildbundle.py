"""FuryNo5 delivery receipt bound to all execution-relevant source contents."""
import hashlib
import json
from pathlib import Path
from .package import PackageError

NAME='build_bundle.json'
def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def sha(data):return hashlib.sha256(data).hexdigest()
def semantic_manifest(manifest):
    return {k:manifest.get(k) for k in ('model','runtime','compatibility','io_contract','memory','state_contract')}
def validate(root,manifest):
    root=Path(root);path=root/NAME
    if not path.exists():return None
    if path.stat().st_size>16*1024**2:raise PackageError('Oversized BuildBundle')
    bundle=json.loads(path.read_text());files=bundle.get('content',{})
    if bundle.get('schema')!='tplmachine.build-bundle.v1' or not files:
        raise PackageError('Invalid BuildBundle')
    canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if sha(canonical(semantic_manifest(manifest)))!=bundle.get('manifest_semantics_sha256'):
        raise PackageError('BuildBundle manifest semantics changed')
    actual={str(p.relative_to(root)):file_sha(p) for p in root.rglob('*') if p.is_file() and str(p.relative_to(root)) not in (NAME,'manifest.json')}
    if actual!=files:raise PackageError('BuildBundle content changed or missing files')
    acceptance=bundle.get('acceptance',{})
    expected=sha(canonical({'files':files,'manifest':bundle['manifest_semantics_sha256']}))
    if bundle.get('content_id')!=expected or acceptance.get('content_id')!=expected:
        raise PackageError('BuildBundle acceptance belongs to different contents')
    if (bundle.get('status')!='validated' or acceptance.get('fresh_process_reload') is not True
            or acceptance.get('numerical_passed') is not True or acceptance.get('completion_confirmed') is not True
            or acceptance.get('state_validation') not in ('passed','not_applicable')
            or type(acceptance.get('npu_tasks')) is not int or acceptance['npu_tasks']<1
            or any(type(acceptance.get(k)) is not int or acceptance[k]!=0 for k in ('cpu_routes','unknown_graphs','compiler_calls'))):
        raise PackageError('BuildBundle has not passed independent execution acceptance')
    stateful=manifest.get('io_contract',{}).get('method') in ('chat','generate') or manifest.get('state_contract') not in (None,{}, {'kind':'stateless'})
    if stateful and acceptance['state_validation']!='passed':
        raise PackageError('BuildBundle stateful model requires state validation')
    return {'content_id':expected,'status':'validated','current_request_npu_execution':'unknown'}
