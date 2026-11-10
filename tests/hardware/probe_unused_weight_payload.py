"""Physical loader-budget probe:8GiB unusedweights plus a qualified small graph.

Distinct GPU targets map to one32MiB checkpoint tensor. This tests residency,
admission and cleanup, not8GiB computation or a distinct9B checkpoint.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from unittest.mock import patch
import zipfile


def memory(label):
    rss=next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
    data=json.loads(subprocess.check_output(['furiosa-smi','status','--format','json'],text=True))
    return {'label':label,'rss_bytes':rss,'npu_bytes':{d['device']:d['memory']['DRAM']['used_size'] for d in data}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fxb',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    os.environ['FXBX_TRACE_STEPS']='1'
    import torch,furiosa.torch as sdk
    from fxbx import GraphFXB,SafeTensorWeights
    from fxbex.execution_fxb import read_execution_plan,validate_execution_plan,export_execution_plan,ExecutionPlanError
    from safetensors.torch import save_file
    from furiosa.torch.profiler import Profiler,RecordFunction
    torch.set_num_threads(2);start=time.monotonic()
    result={'status':'running','scope':'unused replicated8GiB externaltargets; originalsmallEDF performs computation; not9Bweight diversity or IOperformance','memory':[]}
    try:
        plan=read_execution_plan(a.fxb);names={s['target']:s['target'] for s in plan['tensors'].values() if s['kind'] in ('parameter','buffer')}
        original=SafeTensorWeights(a.weights);cpuweights={k:original[k] for k in names}
        payload=torch.zeros(4096,4096,dtype=torch.bfloat16);cpuweights['budget.weight']=payload
        for i in range(256):
            name=f'unused:{i}';target=f'payload_{i:04d}';names[target]='budget.weight'
            plan['tensors'][name]={'kind':'parameter','target':target,'shape':[4096,4096],'stride':[4096,1],'dtype':'bfloat16'}
        plan['memory_policy']={'budget_bytes':16*1024**3,'reserve_bytes':1024**3,'safety_margin_bytes':0}
        plan['source_metadata']={'architecture':'SyntheticUnusedWeightResidency','source':'qualifiedsmallprogram','payload_target_count':256}
        bad=copy.deepcopy(plan);bad['memory_policy']['budget_bytes']=4*1024**3
        with patch.object(sdk,'Runnable',side_effect=AssertionError('budget rejection must precede deviceallocation')) as allocate:
            try:validate_execution_plan(bad,archived=True)
            except ExecutionPlanError as e:
                assert 'budget' in str(e).lower();result['underbudget_rejected_before_allocation']=True
            else:raise AssertionError('4GiBbudget accepted8GiBpayload')
            assert allocate.call_count==0
        extraction=a.out/'edfs';extraction.mkdir()
        with zipfile.ZipFile(a.fxb) as archive:
            for i,k in enumerate(plan['kernels'].values()):
                path=extraction/f'{i}.edf';path.write_bytes(archive.read(k.pop('member')));k['edf_path']=str(path)
        artifact=a.out/'payload.fxb';export_execution_plan(plan,artifact)
        checkpoint=a.out/'checkpoint';checkpoint.mkdir();save_file(cpuweights,str(checkpoint/'payload.safetensors'))
        result.update(payload_gpu_bytes=256*payload.numel()*payload.element_size(),checkpoint_bytes=(checkpoint/'payload.safetensors').stat().st_size)
        result['memory'].append(memory('before_load'));calls=[];trace=a.out/'trace.json'
        with Profiler(str(trace)):
            tick=time.monotonic()
            with GraphFXB(artifact,weights=SafeTensorWeights(checkpoint,names=names)) as runtime:
                result['load_seconds']=time.monotonic()-tick;loaded=memory('loaded');result['memory'].append(loaded)
                assert max(loaded['npu_bytes'].values())>=result['payload_gpu_bytes'],'SMI did not confirm8GiB residentpayload'
                for i in range(2):
                    runtime.reset();x=torch.full((4,8),.25*i,dtype=torch.bfloat16)
                    expected=torch.nn.functional.linear(x,cpuweights['core.weight'],cpuweights['core.bias'])
                    label=f'audit_call::{i}';tick=time.monotonic()
                    with RecordFunction(label,propagate=True):actual,=runtime.run('prefill',(x,))
                    torch.testing.assert_close(actual,expected,atol=.02,rtol=.02);calls.append({'label':label,'host_seconds':time.monotonic()-tick})
            result['memory'].append(memory('closed'))
        events=json.loads(trace.read_text())['traceEvents'];cpu=sum('run_by_cpu' in e.get('name','') for e in events);assert not cpu
        expected={f"execution_plan::prefill::{i}::{s['kernel']}" for i,s in enumerate(plan['entries']['prefill']['steps'])}
        for call in calls:
            frames=[e for e in events if e.get('name')==call['label'] and e.get('ph')=='X'];assert len(frames)==1
            low=frames[0]['ts'];high=low+frames[0]['dur'];window=[e for e in events if 'ts' in e and low<=e['ts'] and e['ts']+e.get('dur',0)<=high]
            spans={e['name'] for e in window if e.get('name','').startswith('execution_plan::')};tasks={e['name'].split('::Task::',1)[1] for e in window if '::Task::execution_plan::' in e.get('name','')}
            assert spans==expected and tasks==expected,'call-boundtraceevidence missing'
        result.update(status='passed',cpu_routes=cpu,calls=calls,call_bounded_steps=len(calls)*len(expected))
    except BaseException as e:
        result.update(status='failed',error_type=type(e).__name__,error=str(e));(a.out/'traceback.txt').write_text(traceback.format_exc())
    result['seconds']=time.monotonic()-start;(a.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['status']=='passed' else 1)


if __name__=='__main__':main()
