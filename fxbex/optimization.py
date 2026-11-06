"""Optional, integrity-bound selection/acceptance receipts, never executable variants."""
import hashlib
import json
from pathlib import Path


def specification(manifest):
    from .package import PackageError
    spec=manifest.get('optimization')
    if spec is None:return None
    if (not isinstance(spec,dict) or type(spec.get('schema_version')) is not int or spec['schema_version']!=1
            or spec.get('backend') not in ('torch_runnable','furiosa_opt_native')
            or spec.get('artifact_format') != {'torch_runnable':'sdk-runnable',
                'furiosa_opt_native':'native-image'}[spec['backend']]):
        raise PackageError('Invalid optimization backend/artifact contract')
    adapter=manifest['runtime']['adapter']
    if spec['backend']=='furiosa_opt_native' and adapter in ('qwen35_replay','runnable_dag'):
        raise PackageError('Native images cannot replace torch Runnable artifacts')
    for key in ('selection','acceptance'):
        if not isinstance(spec.get(key),str) or spec[key] not in manifest['files']:
            raise PackageError('Optimization evidence must be in package integrity map')
    return spec


def validate_evidence(root,manifest):
    from .package import PackageError,DEFAULT_MAX_JSON_SIZE,_json
    spec=specification(manifest)
    if spec is None:return None
    docs={}
    for key in ('selection','acceptance'):
        path=(Path(root)/spec[key]).resolve()
        if not path.is_relative_to(Path(root).resolve()) or path.stat().st_size>DEFAULT_MAX_JSON_SIZE:
            raise PackageError('Unsafe or oversized optimization evidence')
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=manifest['files'][spec[key]]['sha256']:
            raise PackageError('Optimization evidence checksum mismatch')
        docs[key]=_json(data,spec[key])
    acceptance=docs['acceptance']
    selection=docs['selection']
    if (not isinstance(selection,dict) or selection.get('schema') not in (
            'fha.tplmachine-campaign.v1','tplmachine.measured-selection.v1')
            or selection.get('deployable') is not False):
        raise PackageError('Optimization selection must remain an advisory record')
    compatibility=hashlib.sha256(json.dumps(manifest['compatibility'],sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (acceptance.get('schema')!='fxbex.model-acceptance.v1'
            or acceptance.get('backend')!=spec['backend']
            or acceptance.get('execution_plan_sha256')!=manifest['files']['execution_plan.json']['sha256']
            or acceptance.get('compatibility_sha256')!=compatibility
            or acceptance.get('numerical_passed') is not True
            or acceptance.get('state_validation') not in ('passed','not_applicable')
            or (manifest.get('io_contract',{}).get('method')=='chat' and acceptance.get('state_validation')!='passed')
            or acceptance.get('completion_confirmed') is not True
            or type(acceptance.get('npu_tasks')) is not int or acceptance['npu_tasks']<=0
            or any(type(acceptance.get(k)) is not int or acceptance[k]!=0 for k in (
                'cpu_routes','unknown_graphs','compiler_calls'))):
        raise PackageError('Optimization model acceptance incomplete or bound to another plan/environment')
    return {'selection_present':True,'acceptance_bound':True,
            'runtime_action':'none','current_request_npu_execution':'unknown'}
