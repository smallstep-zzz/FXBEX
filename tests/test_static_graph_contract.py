"""Offline metadata checks; fake hashes and ABI text are not NPU evidence."""
import copy
import unittest

from fxbex.graph_fxb import GraphFXBError
from fxbex.static_graph_contract import StaticGraphContractProvider


class StaticGraphContractTests(unittest.TestCase):
    def setUp(self):
        self.provider = StaticGraphContractProvider()
        self.metadata = {'architecture': 'torch.nn.Linear', 'in_features': 8,
                         'out_features': 6, 'bias': True, 'dtype': 'bfloat16'}
        self.export = {
            'source_identity': {'graph_sha256': 'a' * 64, 'graph_code_sha256': 'b' * 64},
            'edf_sha256': 'c' * 64, 'expected_num_fusion': 8, 'target_npu': 'renegade',
            'inputs': [
                {'name': 'p_weight', 'kind': 'parameter', 'target': 'weight',
                 'shape': [6, 8], 'stride': [8, 1], 'dtype': 'bfloat16'},
                {'name': 'x', 'kind': 'user_input', 'user_index': 0,
                 'shape': [4, 8], 'stride': [8, 1], 'dtype': 'bfloat16'},
            ],
            'outputs': [{'name': 'y', 'shape': [4, 6], 'stride': [6, 1], 'dtype': 'bfloat16'}],
            'abi': {'inputs': ['weight ABI', 'x ABI'], 'outputs': ['y ABI']},
            'compatibility': {'torch': '2.10.0+cpu', 'furiosa-torch': '2026.4.0',
                              'furiosa-tcc': '2026.4.0'},
            'compiler_identity': {'version': '2026.4.0', 'revision': 'fixture-revision'},
            'compiler_config': {'test_option': True},
        }

    def provide(self, export=None, metadata=None):
        return self.provider.provide(model_metadata=self.metadata if metadata is None else metadata,
                                     edf_export=self.export if export is None else export)

    def fingerprint(self, contract):
        return contract['source']['fxbx_graph_fingerprint']

    def test_truthful_manifest_and_explicit_consumer(self):
        result = self.provide()
        manifest = result['manifest']
        self.assertEqual(manifest['model']['architecture'], 'torch.nn.Linear')
        self.assertEqual(manifest['versions'], {'furiosa_compiler': self.export['compiler_identity']})
        self.assertNotIn('furiosa_llm', manifest['versions'])
        self.assertEqual(result['bucket'], {'tokenwise': {'input_size': 4}})
        self.assertEqual(manifest['kernels']['forward'],
                         [{'bucket': result['bucket'], 'file': result['member']}])
        self.assertEqual(result['device_config'], {'fusion': 8, 'mesh': 'single'})
        self.assertIs(result['source']['official_llm_dispatch'], False)
        self.assertIs(result['source']['explicit_consumer_required'], True)

    def test_fingerprint_is_deterministic_but_archive_identity_is_fresh(self):
        first, second = self.provide(), self.provide()
        self.assertEqual(self.fingerprint(first), self.fingerprint(second))
        self.assertNotEqual(first['manifest']['uuid'], second['manifest']['uuid'])

    def test_graph_abi_edf_sdk_config_and_model_changes_invalidate_identity(self):
        baseline = self.fingerprint(self.provide())
        variants = []
        for key, value in [('edf_sha256', 'd' * 64), ('compiler_config', None)]:
            changed = copy.deepcopy(self.export); changed[key] = value; variants.append(changed)
        changed = copy.deepcopy(self.export); changed['source_identity']['graph_sha256'] = 'e' * 64; variants.append(changed)
        changed = copy.deepcopy(self.export); changed['abi']['outputs'][0] = 'different ABI'; variants.append(changed)
        changed = copy.deepcopy(self.export); changed['compatibility']['torch'] = '2.10.1'; variants.append(changed)
        changed = copy.deepcopy(self.export); changed['compiler_identity']['revision'] = 'new-revision'; variants.append(changed)
        for changed in variants:
            self.assertNotEqual(self.fingerprint(self.provide(changed)), baseline)
        metadata = dict(self.metadata, bias=False)
        self.assertNotEqual(self.fingerprint(self.provide(metadata=metadata)), baseline)

    def test_missing_or_malformed_compiler_source_and_device_contracts_rejected(self):
        for key, value in [('source_identity', {}), ('edf_sha256', 'not-a-hash'),
                           ('expected_num_fusion', 4), ('expected_num_fusion', True),
                           ('target_npu', 'warboy'), ('compatibility', {}),
                           ('compiler_identity', {}), ('compiler_config', 'guessed')]:
            changed = copy.deepcopy(self.export); changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(GraphFXBError):
                self.provide(changed)
        for metadata in ({}, {'architecture': ''}, {'architecture': 'x\nspoof'}):
            with self.assertRaises(GraphFXBError): self.provide(metadata=metadata)
        changed = copy.deepcopy(self.export); changed.pop('compiler_config')
        with self.assertRaises(GraphFXBError): self.provide(changed)

    def test_static_binding_and_finite_json_guards(self):
        for mutation in ('zero', 'scalar_user', 'missing_abi', 'duplicate_user', 'nan'):
            changed = copy.deepcopy(self.export)
            if mutation == 'zero': changed['inputs'][1]['shape'][0] = 0
            elif mutation == 'scalar_user': changed['inputs'][1].update(shape=[], stride=[])
            elif mutation == 'missing_abi': changed['abi']['inputs'].pop()
            elif mutation == 'duplicate_user': changed['inputs'][0].update(kind='user_input', user_index=0)
            else: changed['compiler_config']['bad'] = float('nan')
            with self.subTest(mutation=mutation), self.assertRaises(GraphFXBError): self.provide(changed)


if __name__ == '__main__':
    unittest.main()
