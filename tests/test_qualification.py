"""Qualification identity tests use CPU checkpoint data, never an NPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fxbex.checkpoint import SafeTensorWeights
from fxbex.qualification import (SCHEMA, QualificationError, qualification_bindings,
                                 validate_qualification)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import save_file
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fxb = self.root / 'model.fxb'; self.fxb.write_bytes(b'fixture-edf-archive')
        self.checkpoint = self.root / 'weights'; self.checkpoint.mkdir()
        save_file({'w': torch.ones(2), 'other': torch.zeros(2)}, str(self.checkpoint / 'model.safetensors'))
        self.index = self.checkpoint / 'model.safetensors.index.json'
        self.index.write_text(json.dumps({'weight_map': {'w': 'model.safetensors', 'other': 'model.safetensors'}}))
        self.weights = SafeTensorWeights(self.checkpoint, names={'weight': 'w'})
        self.plan = {'schema': 'tclmachine.execution-plan.v1', 'compiler_identity': {'version': 'test', 'revision': 'test'},
                     'device_config': {'fusion': 8, 'mesh': 'single'},
                     'tensors': {'w': {'kind': 'parameter', 'target': 'weight'}},
                     'entries': {'forward': {}}}
        versions = patch('fxbex.qualification.importlib.metadata.version', return_value='test')
        identity = patch('fxbex.qualification.consumer_identity', return_value={'name': 'fxbx', 'source_sha256': 'a' * 64})
        versions.start(); identity.start(); self.addCleanup(versions.stop); self.addCleanup(identity.stop)
        self.receipt = {'schema': SCHEMA, 'status': 'verified', 'route': 'tcl_static',
                        'bindings': qualification_bindings(self.fxb, self.plan, self.weights),
                        'scope': {'kind': 'fixed_shape_fixtures', 'entries': ['forward'], 'output_devices': ['cpu'], 'atol': .02, 'rtol': .02},
                        'evidence': {'call_count': 3, 'cpu_routes': 0, 'fixture_readonly_checks': True}}
        self.path = self.root / 'qualification.json'

    def validate(self, receipt=None, *, plan=None, weights=None):
        self.path.write_text(json.dumps(self.receipt if receipt is None else receipt))
        return validate_qualification(self.fxb, self.plan if plan is None else plan,
                                      self.weights if weights is None else weights, self.path)

    def test_matching_receipt_and_nonexecuting_data(self):
        result = self.validate()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['scope']['entries'], ['forward'])
        self.assertEqual(self.weights.loaded_tensors, 0)
        self.assertEqual(len(result['receipt_sha256']), 64)

    def test_changed_fxb_or_plan_is_rejected(self):
        changed = copy.deepcopy(self.plan); changed['compiler_identity']['revision'] = 'new'
        with self.assertRaisesRegex(QualificationError, 'does not match'):self.validate(plan=changed)
        self.fxb.write_bytes(b'changed')
        with self.assertRaisesRegex(QualificationError, 'does not match'):self.validate()

    def test_same_shape_different_weight_values_are_not_qualified(self):
        import torch
        from safetensors.torch import save_file
        save_file({'w': torch.zeros(2), 'other': torch.zeros(2)}, str(self.checkpoint / 'model.safetensors'))
        new = SafeTensorWeights(self.checkpoint, names={'weight': 'w'})
        with self.assertRaisesRegex(QualificationError, 'does not match'):self.validate(weights=new)

    def test_names_mapping_and_index_are_bound(self):
        new = SafeTensorWeights(self.checkpoint, names={'weight': 'other'})
        with self.assertRaisesRegex(QualificationError, 'does not match'):self.validate(weights=new)
        self.index.write_text(self.index.read_text() + '\n')
        with self.assertRaisesRegex(QualificationError, 'does not match'):self.validate()

    def test_dict_weights_cannot_inherit_checkpoint_qualification(self):
        with self.assertRaisesRegex(QualificationError, 'SafeTensorWeights'):self.validate(weights={'weight': object()})

    def test_status_route_schema_scope_and_evidence_fail_closed(self):
        changes = [lambda r:r.update(status='building'), lambda r:r.update(schema='future'),
                   lambda r:r.update(route='official_llm'), lambda r:r['scope'].update(entries=['decode']),
                   lambda r:r['scope'].update(atol=float('nan')), lambda r:r['evidence'].update(cpu_routes=True),
                   lambda r:r['evidence'].update(fixture_readonly_checks=False), lambda r:r['evidence'].update(call_count=0),
                   lambda r:r['bindings']['consumer'].update(source_sha256='b' * 64),
                   lambda r:r['bindings']['environment']['packages'].update(torch='new')]
        for change in changes:
            with self.subTest(change=change):
                receipt = copy.deepcopy(self.receipt); change(receipt)
                with self.assertRaises(QualificationError):self.validate(receipt)

    def test_duplicate_and_oversized_receipts_rejected(self):
        self.path.write_text('{"status":"failed","status":"verified"}')
        with self.assertRaisesRegex(QualificationError, 'Duplicate'):
            validate_qualification(self.fxb, self.plan, self.weights, self.path)
        self.path.write_bytes(b' ' * (4 * 1024**2 + 1))
        with self.assertRaisesRegex(QualificationError, 'oversized'):
            validate_qualification(self.fxb, self.plan, self.weights, self.path)

    def test_no_external_weights_need_no_dummy_checkpoint(self):
        self.plan['tensors'] = {}
        receipt = copy.deepcopy(self.receipt)
        receipt['bindings'] = qualification_bindings(self.fxb, self.plan, {})
        self.assertEqual(self.validate(receipt, weights={})['status'], 'verified')


if __name__ == '__main__':unittest.main()
