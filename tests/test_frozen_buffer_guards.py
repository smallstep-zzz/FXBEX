"""Offline frozen-specialization guards; fake SDK never establishes NPU evidence."""
import copy
import hashlib
import struct
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

if __package__:
    from . import test_execution_fxb as fixtures
else:
    import test_execution_fxb as fixtures
from fxbex.execution_fxb import (
    ExecutionFXB, ExecutionPlanError, frozen_buffer_sha256, read_execution_plan,
    validate_execution_plan,
)

try:
    import torch
except ImportError:
    torch = None


class FrozenBufferPlanTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ExecutionPlanTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.plan['schema']='tclmachine.execution-plan.v2'
        self.fixture.plan['tensors']['positions']={
            'shape':[1,8],'stride':[8,1],'dtype':'int64','kind':'buffer','target':'position_ids'}
        self.fixture.plan['frozen_buffer_guards']=[{
            'target':'position_ids','shape':[1,8],'dtype':'int64','sha256':'0'*64}]

    def test_v2_round_trips_guards_in_native_archive(self):
        self.fixture.export()
        actual=read_execution_plan(self.fixture.output)
        self.assertEqual(actual['schema'],'tclmachine.execution-plan.v2')
        self.assertEqual(actual['frozen_buffer_guards'],self.fixture.plan['frozen_buffer_guards'])
        self.assertEqual(actual['tensors']['positions']['kind'],'buffer')

    def test_schema_version_requires_guards_only_for_v2(self):
        for schema,guards in (('tclmachine.execution-plan.v1',[]),
                              ('tclmachine.execution-plan.v1',self.fixture.plan['frozen_buffer_guards']),
                              ('tclmachine.execution-plan.v2',[]),
                              ('tclmachine.execution-plan.v2',None),
                              ('tclmachine.execution-plan.v2',{})):
            plan=copy.deepcopy(self.fixture.plan);plan['schema']=schema;plan['frozen_buffer_guards']=guards
            with self.subTest(schema=schema,guards=guards),self.assertRaises(ExecutionPlanError):
                validate_execution_plan(plan)
        plan=copy.deepcopy(self.fixture.plan);plan.pop('frozen_buffer_guards')
        with self.assertRaises(ExecutionPlanError):validate_execution_plan(plan)
        plan['schema']='tclmachine.execution-plan.v1'
        validate_execution_plan(plan)

    def test_guard_fields_must_match_a_declared_buffer_exactly(self):
        for change in ({'target':'unknown'},{'target':False},{'shape':[8]},
                       {'shape':[1,False]},{'shape':[True,8]},{'dtype':'int32'},{'dtype':None},
                       {'sha256':'a'*63},{'sha256':'A'*64},{'sha256':'g'*64},
                       {'sha256':None},{'extra':1}):
            plan=copy.deepcopy(self.fixture.plan);plan['frozen_buffer_guards'][0].update(change)
            with self.subTest(change=change),self.assertRaises(ExecutionPlanError):
                validate_execution_plan(plan)
        for kind in ('parameter','state','intermediate'):
            plan=copy.deepcopy(self.fixture.plan);plan['tensors']['positions']['kind']=kind
            if kind=='state':plan['tensors']['positions']['initial']='zeros'
            with self.subTest(kind=kind),self.assertRaises(ExecutionPlanError):validate_execution_plan(plan)
        for guards in ([None],[[]],[{}],self.fixture.plan['frozen_buffer_guards']*2):
            plan=copy.deepcopy(self.fixture.plan);plan['frozen_buffer_guards']=guards
            with self.assertRaises(ExecutionPlanError):validate_execution_plan(plan)

    def test_per_buffer_and_total_guard_byte_limits(self):
        plan=copy.deepcopy(self.fixture.plan)
        plan['tensors']['positions'].update(shape=[8192],stride=[1])
        plan['frozen_buffer_guards'][0]['shape']=[8192]
        validate_execution_plan(plan)
        plan['tensors']['positions']['shape']=[8193];plan['frozen_buffer_guards'][0]['shape']=[8193]
        with self.assertRaisesRegex(ExecutionPlanError,'64 KiB'):validate_execution_plan(plan)
        plan=copy.deepcopy(self.fixture.plan);plan['frozen_buffer_guards']=[]
        for index in range(17):
            target=f'buffer_{index}'
            plan['tensors'][target]={'shape':[8192],'stride':[1],'dtype':'int64','kind':'buffer','target':target}
            plan['frozen_buffer_guards'].append({'target':target,'shape':[8192],'dtype':'int64','sha256':'0'*64})
            if index==15:validate_execution_plan(plan)
        with self.assertRaisesRegex(ExecutionPlanError,'1 MiB'):validate_execution_plan(plan)


@unittest.skipIf(torch is None,'optional PyTorch')
class FrozenBufferValueTests(unittest.TestCase):
    def setUp(self):
        fixture=FrozenBufferPlanTests();fixture.setUp()
        self.addCleanup(fixture.doCleanups);self.fixture=fixture.fixture
        self.value=torch.arange(8,dtype=torch.int64).reshape(1,8)
        self.fixture.plan['frozen_buffer_guards'][0]['sha256']=frozen_buffer_sha256(self.value)

    def test_canonical_hash_is_shape_dtype_and_raw_value_sensitive(self):
        before=self.value.clone()
        expected=hashlib.sha256(b'int64\n[1,8]\n'+struct.pack('<8q',*range(8))).hexdigest()
        self.assertEqual(frozen_buffer_sha256(self.value),expected)
        self.assertTrue(torch.equal(self.value,before))
        for other in (self.value.reshape(8),self.value.to(torch.int32),self.value+1):
            self.assertNotEqual(frozen_buffer_sha256(other),expected)
        transposed=self.value.reshape(2,4).t()
        self.assertEqual(frozen_buffer_sha256(transposed),frozen_buffer_sha256(transposed.contiguous()))

    def test_hash_rejects_device_grad_oversize_and_unsupported_values(self):
        for value in (None,[],torch.ones(2,requires_grad=True),torch.ones(65537,dtype=torch.uint8),
                      torch.empty(2,device='meta'),torch.empty(0),torch.ones(2,dtype=torch.complex64),
                      torch.nn.Parameter(torch.ones(2),requires_grad=False)):
            with self.subTest(value_type=type(value)),self.assertRaises(ExecutionPlanError):
                frozen_buffer_sha256(value)
        self.assertEqual(len(frozen_buffer_sha256(torch.tensor(True))),64)
        self.assertEqual(len(frozen_buffer_sha256(torch.ones(2,dtype=torch.bfloat16))),64)

    def _assert_load_guard(self,weights,*,mismatch):
        self.fixture.export();modules,calls=self.fixture.backend();modules.pop('torch')
        marker=RuntimeError('CPU guard admitted before SDK loading')
        modules['furiosa.torch'].set_fusion=lambda _:(_ for _ in ()).throw(marker)
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'):
            with self.assertRaisesRegex(ExecutionPlanError if mismatch else RuntimeError,
                                        'Frozen buffer hash mismatch' if mismatch else str(marker)):
                with ExecutionFXB(self.fixture.output,weights=weights):pass
        self.assertFalse(any(c[0] in ('load','preprocess','run') for c in calls))

    def test_dictionary_guard_mismatch_stops_before_runnable_loading(self):
        self._assert_load_guard({'position_ids':self.value+1},mismatch=True)

    def test_dictionary_guard_success_reaches_sdk_loading(self):
        self._assert_load_guard({'position_ids':self.value},mismatch=False)

    def test_safetensors_guard_checks_cpu_payload_before_runnable_loading(self):
        try:
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest('optional safetensors')
        from fxbex.checkpoint import SafeTensorWeights
        for mismatch in (False,True):
            with self.subTest(mismatch=mismatch),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'buffer.safetensors'
                save_file({'position_ids':self.value+int(mismatch)},str(path))
                weights=SafeTensorWeights(path)
                # Each subcase needs its own output archive.
                self.fixture.output=self.fixture.root/f'guard-{mismatch}.fxb'
                self._assert_load_guard(weights,mismatch=mismatch)
                self.assertEqual(weights.loaded_tensors,1)


if __name__=='__main__':unittest.main()
