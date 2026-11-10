"""Reuse a qualified one-EDF plan through legacy graph transport on real NPU."""
import argparse,json,time,zipfile,hashlib,traceback
from pathlib import Path
import torch
from safetensors.torch import load_file
from fxbx import GraphFXB,SafeTensorWeights
from fxbex.graph_fxb import ExistingNativeContract,export_graph
from fxbex.execution_fxb import read_execution_plan
from furiosa.torch.profiler import Profiler

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fxb',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
 root=a.out;root.mkdir(parents=True,exist_ok=False)
 source=a.fxb;plan=read_execution_plan(source);entry=plan['entries']['forward']
 assert len(entry['steps'])==1 and len(plan['kernels'])==1
 name,kernel=next(iter(plan['kernels'].items()))
 with zipfile.ZipFile(source) as archive:
  manifest=json.loads(archive.read('manifest.json'));data=archive.read(kernel['member'])
 edf=root/'qualified.edf';edf.write_bytes(data)
 inputs=[];index=0
 for tid in entry['steps'][0]['inputs']:
  s=dict(plan['tensors'][tid],name=tid)
  if s['kind']=='user_input':s['user_index']=index;index+=1
  inputs.append(s)
 export={'edf_path':str(edf),'edf_sha256':kernel['edf_sha256'],'abi':kernel['abi'],'inputs':inputs,
         'outputs':[dict(plan['tensors'][tid],name=tid) for tid in entry['outputs']],
         'compatibility':plan['compatibility'],'expected_num_fusion':8,'validation':{'qualified_source':str(source)}}
 bucket=manifest['kernels'][name][0]['bucket'];metadata=plan['source_metadata']
 provider=ExistingNativeContract(manifest,metadata,name,bucket,kernel['member'],{'kind':'qualified-existing-single-EDF'},device_config=plan['device_config'])
 result={'status':'running','source_fxb_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'qualified single-EDF replay, not model-architecture registration'}
 try:
  export_graph(export,root/'legacy.fxb',model_metadata=metadata,contract_provider=provider)
  evidence=Path(str(source)+'.verified');spec=json.loads((evidence/'replay-spec.json').read_text())
  weights=dict(SafeTensorWeights(evidence/'weights'));args=tuple(load_file(str(evidence/f))['value'] for f in spec['calls'][0]['inputs']);expected=tuple(load_file(str(evidence/f))['value'] for f in spec['calls'][0]['expected'])
  import os;os.environ['FXBX_TRACE_STEPS']='1'
  started=time.monotonic()
  with Profiler(str(root/'trace.json')):
   with GraphFXB(root/'legacy.fxb',weights=weights) as runtime:
    assert type(runtime).__name__=='GraphFXB'
    for i in range(10):
     actual=runtime.predict(args,output_device='furiosa' if i%2 else 'cpu')
     for a,b in zip(actual,expected):
      assert a.device.type==('furiosa' if i%2 else 'cpu')
      torch.testing.assert_close(a.cpu(),b,atol=.02,rtol=.02)
  events=json.loads((root/'trace.json').read_text())['traceEvents'];tasks=[e for e in events if '::Task' in e.get('name','')];cpu=sum('run_by_cpu' in e.get('name','') for e in events)
  assert tasks and not cpu
  result.update(status='passed',calls=10,task_events=len(tasks),cpu_routes=cpu,seconds=time.monotonic()-started)
 except Exception as e:
  result.update(status='failed',error=type(e).__name__+': '+str(e));(root/'traceback.txt').write_text(traceback.format_exc())
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
