import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from fxbex.optimization import specification,validate_evidence
from fxbex.package import PackageError


class Optimization(unittest.TestCase):
    def test_bound_acceptance_is_not_current_request_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            compatibility={'hardware':'rngd','fusion':8}
            acceptance=dict(schema='fxbex.model-acceptance.v1',backend='torch_runnable',
                execution_plan_sha256='a'*64,compatibility_sha256=hashlib.sha256(json.dumps(compatibility,
                    sort_keys=True,separators=(',',':')).encode()).hexdigest(),numerical_passed=True,
                state_validation='passed',completion_confirmed=True,npu_tasks=10,cpu_routes=0,
                unknown_graphs=0,compiler_calls=0)
            selection={'schema':'tplmachine.measured-selection.v1','deployable':False}
            manifest={'runtime':{'adapter':'qwen35_replay'},'compatibility':compatibility,
                'io_contract':{'method':'chat'},'files':{'execution_plan.json':{'sha256':'a'*64}},
                'optimization':{'schema_version':1,'backend':'torch_runnable','artifact_format':'sdk-runnable',
                                'selection':'selection.json','acceptance':'acceptance.json'}}
            def write():
                for name,data in [('selection.json',selection),('acceptance.json',acceptance)]:
                    raw=json.dumps(data).encode();(root/name).write_bytes(raw)
                    manifest['files'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)}
            write()
            self.assertEqual(validate_evidence(root,manifest)['current_request_npu_execution'],'unknown')
            for key,value in [('execution_plan_sha256','b'*64),('compiler_calls',1),('unknown_graphs',1),
                              ('state_validation','not_applicable')]:
                original=acceptance[key];acceptance[key]=value;write()
                with self.assertRaises(PackageError):validate_evidence(root,manifest)
                acceptance[key]=original
            manifest['optimization'].update(backend='furiosa_opt_native',artifact_format='native-image')
            with self.assertRaisesRegex(PackageError,'cannot replace'):specification(manifest)

    def test_legacy_package_has_no_new_requirement(self):
        self.assertIsNone(validate_evidence(None,{}))
