import gc
import json
from pathlib import Path
import tempfile
import unittest
import weakref

from fxbex.memory import (TensorLifetime, analyze, dag_plan, stateful_plan,
                          tensor_bytes, inspect_archive)
from fxbex.package import PackageError, build


SPEC = {'shape': [1024], 'dtype': 'bfloat16'}


def fixture():
    execution = {'format': 'runnable-dag-v1', 'steps': [
        {'id': 'a', 'artifact': 'a.edf', 'inputs': ['x', 'w'], 'outputs': ['a']},
        {'id': 'b', 'artifact': 'a.edf', 'inputs': ['a', 'a'], 'outputs': ['b', 'unused']},
        {'id': 'c', 'artifact': 'a.edf', 'inputs': ['a', 'b'], 'outputs': ['y']}]}
    config = {'inputs': {'x': SPEC}, 'outputs': ['y'],
              'tensor_specs': {n: SPEC for n in ['a', 'b', 'unused', 'y']}}
    index = {'weights': [{'path': 'w.pt', 'tensors': {'w': SPEC}}]}
    return execution, config, index


class FakeTensor:
    def numel(self): return 1024
    def element_size(self): return 2


class MemoryTests(unittest.TestCase):
    def test_branch_repeated_input_dead_output_and_final_output(self):
        plan = dag_plan(*fixture(), {'policy': {'release_threshold_bytes': 0}})
        self.assertEqual(plan['release_after_step'], [['x'], ['unused'], ['a', 'b']])
        self.assertEqual(plan['last_use']['y'], 3)
        self.assertEqual(plan['estimate']['peak_live_tensor_logical_bytes'], 3 * 2048)
        self.assertEqual(plan['estimate']['retained_until_return_tensor_logical_bytes'], 5 * 2048)
        self.assertEqual(plan['estimate']['weight_logical_bytes'], 2048)
        self.assertIsNone(plan['estimate']['physical_allocated_bytes'])

    def test_completion_precedes_release_and_aliases_remain_valid(self):
        plan = dag_plan(*fixture(), {'policy': {'release_threshold_bytes': 0}})
        values = {'x': FakeTensor(), 'w': FakeTensor(), 'a': FakeTensor()}
        reference = weakref.ref(values['x'])
        barriers = []
        def complete():
            self.assertIn('x', values)
            self.assertIsNotNone(reference())
            barriers.append(1)
        lifetime = TensorLifetime(plan, {'w'}, complete)
        lifetime.after_step(0, values)
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(len(barriers), 1)
        values['b'] = values['a']  # A public view/alias must not be manually overwritten.
        values['unused'] = FakeTensor()
        lifetime.synchronize = lambda: None
        lifetime.after_step(1, values)
        values['y'] = values['a']
        lifetime.after_step(2, values)
        self.assertEqual(set(values), {'w', 'y'})
        self.assertEqual(lifetime.summary()['released_tensor_bindings'], 4)

    def test_unknown_completion_retains_legacy_lifetimes(self):
        plan = dag_plan(*fixture(), {'policy': {'release_threshold_bytes': 0}})
        values = {'x': FakeTensor(), 'a': FakeTensor()}
        life = TensorLifetime(plan, {}, None)
        life.after_step(0, values)
        self.assertIn('x', values)
        self.assertIn('legacy_retention', life.summary()['release_policy'])

    def test_batched_barriers_have_their_own_peak_estimate(self):
        plan = dag_plan(*fixture())
        self.assertEqual(plan['estimate']['planned_batched_peak_tensor_logical_bytes'], 5 * 2048)
        self.assertEqual(plan['estimate']['known_logical_peak_bytes'], 6 * 2048)

    def test_stale_plan_is_rejected(self):
        execution, config, index = fixture()
        plan = dag_plan(execution, config, index)
        config['inputs']['x'] = {'shape': [2048], 'dtype': 'bfloat16'}
        with self.assertRaisesRegex(PackageError, 'Stale memory plan'):
            dag_plan(execution, config, index, plan)

    def test_dynamic_or_missing_metadata_stays_unknown(self):
        execution, config, index = fixture()
        config['tensor_specs']['a'] = {'shape': ['batch', 1024], 'dtype': 'float32'}
        plan = dag_plan(execution, config, index)
        self.assertIn('a', plan['estimate']['unknown_tensors'])
        self.assertIsNone(tensor_bytes({'shape': [True], 'dtype': 'float32'}))
        self.assertEqual(tensor_bytes({'shape': [], 'dtype': 'float32'}), 4)
        self.assertIsNone(tensor_bytes({'shape': [1], 'dtype': 'unsupported'}))

    def test_heterogeneous_paged_and_recurrent_states_and_budget(self):
        states = [{'name': 'kv', 'shape': [1, 4, 512, 256], 'dtype': 'bfloat16',
                   'resident_copies': 0, 'maximum_copies': 2 * 8 * 512},
                  {'name': 'recurrent', 'shape': [16, 32, 32], 'dtype': 'float32',
                   'resident_copies': 24, 'maximum_copies': 24}]
        plan = stateful_plan(10, states)
        self.assertEqual(plan['estimate']['persistent_maximum_logical_bytes'], 8 * 1024**3 + 24 * 65536)
        self.assertEqual(plan['estimate']['persistent_current_logical_bytes'], 24 * 65536)
        with self.assertRaisesRegex(PackageError, 'configured budget'):
            stateful_plan(10, states, policy={'budget_bytes': 1024})

    def test_build_includes_recomputed_plan_and_small_metadata_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            root.mkdir()
            execution, config, index = fixture()
            index['weights'] = []
            execution['steps'][0]['inputs'] = ['x']
            documents = {'execution_plan.json': execution, 'runtime_config.json': config,
                         'weights_index.json': index, 'memory_plan.json': {},
                         'manifest.json': {'model': {'name': 'test'}, 'runtime': {'adapter': 'runnable_dag'}, 'compatibility': {}}}
            for name, value in documents.items():
                (root / name).write_text(json.dumps(value))
            (root / 'a.edf').write_bytes(b'fixture, not an NPU executable')
            package = build(root, Path(directory) / 'model.fxbex')
            result = inspect_archive(package)
            self.assertEqual(result['kind'], 'dag-logical-memory-v1')
            self.assertIn('metadata only', result['integrity_scope'])
            self.assertEqual(analyze(root)['last_use'], result['last_use'])
            (root / 'memory_plan.json').write_text(json.dumps({'policy': {'budget_bytes': 1}}))
            from fxbex.runtime import RunnableDAG
            with self.assertRaisesRegex(PackageError, 'configured budget'):
                RunnableDAG(root)  # Reject before SDK import, kernel load or weight upload.
