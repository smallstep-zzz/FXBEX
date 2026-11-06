import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from fxbex.buildbundle import validate
from fxbex.package import PackageError,build,extract

canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
sha=lambda b:hashlib.sha256(b).hexdigest()
class Bundle(unittest.TestCase):
 def test_all_contents_bound_and_unvalidated_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'source';root.mkdir();manifest={'model':{'name':'tiny'},'runtime':{'adapter':'test'},'compatibility':{}}
   (root/'manifest.json').write_text(json.dumps(manifest))
   for name in ['execution_plan.json','weights_index.json','memory_plan.json','runtime_config.json']:
    (root/name).write_text('{"weights":[]}' if name=='weights_index.json' else '{}')
   files={p.name:sha(p.read_bytes()) for p in root.iterdir() if p.name!='manifest.json'}
   semantics={k:manifest.get(k) for k in ('model','runtime','compatibility','io_contract','memory','state_contract')}
   content=sha(canonical({'files':files,'manifest':sha(canonical(semantics))}))
   receipt={'schema':'tplmachine.build-bundle.v1','status':'validated','content':files,'manifest_semantics_sha256':sha(canonical(semantics)),'content_id':content,
            'acceptance':dict(content_id=content,fresh_process_reload=True,numerical_passed=True,completion_confirmed=True,state_validation='not_applicable',npu_tasks=1,cpu_routes=0,unknown_graphs=0,compiler_calls=0)}
   (root/'build_bundle.json').write_text(json.dumps(receipt));self.assertEqual(validate(root,manifest)['status'],'validated')
   output=Path(d)/'tiny.fxbex';build(root,output)
   with extract(output) as extracted:self.assertEqual(validate(extracted,json.loads((extracted/'manifest.json').read_text()))['status'],'validated')
   (root/'runtime_config.json').write_text('{"changed":true}')
   with self.assertRaisesRegex(PackageError,'content changed'):validate(root,manifest)
   (root/'runtime_config.json').write_text('{}');receipt['acceptance']['fresh_process_reload']=False
   (root/'build_bundle.json').write_text(json.dumps(receipt))
   with self.assertRaisesRegex(PackageError,'acceptance'):validate(root,manifest)
