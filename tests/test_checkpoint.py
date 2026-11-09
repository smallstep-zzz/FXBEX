import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file
from fxbx import SafeTensorWeights


def spec(shape, dtype='float32'):
    size = 1; strides = []
    for n in reversed(shape):
        strides.insert(0, size); size *= max(n, 1)
    return {'shape': shape, 'dtype': dtype, 'stride': strides}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_header_only_shards_and_explicit_subset_have_no_value_reads(self):
        save_file({'source.w': np.ones((2, 3), dtype=np.float32)}, str(self.root / 'a.safetensors'))
        save_file({'source.b': np.ones(2, dtype=np.float32)}, str(self.root / 'b.safetensors'))
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
            'source.w': 'a.safetensors', 'source.b': 'b.safetensors'}}))
        values = SafeTensorWeights(self.root, names={'weight': 'source.w'})
        values.validate({'weight': spec([2, 3])})
        self.assertEqual(values.loaded_tensors, 0)
        self.assertEqual(list(values), ['weight'])
        self.assertEqual(tuple(values['weight'].shape), (2, 3))
        self.assertEqual(values.loaded_tensors, 1)

    def test_dtype_shape_and_unused_weights_fail_without_reading_values(self):
        p = self.root / 'weights.safetensors'
        save_file({'w': np.ones((2, 3), dtype=np.float32)}, str(p))
        values = SafeTensorWeights(p)
        for targets in ({'w': spec([3, 2])}, {'w': spec([2, 3], 'bfloat16')}, {}):
            with self.assertRaises(ValueError): values.validate(targets)
        self.assertEqual(values.loaded_tensors, 0)

    def test_duplicate_names_and_index_mismatch_are_refused(self):
        for name in ('a', 'b'):
            save_file({'w': np.ones(2, dtype=np.float32)}, str(self.root / (name + '.safetensors')))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            SafeTensorWeights(self.root)
        (self.root / 'b.safetensors').unlink()
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'wrong': 'a.safetensors'}}))
        with self.assertRaisesRegex(ValueError, 'weight_map'):
            SafeTensorWeights(self.root)

    def test_changed_file_and_unsafe_index_are_refused(self):
        p = self.root / 'a.safetensors'
        save_file({'w': np.ones(2, dtype=np.float32)}, str(p))
        values = SafeTensorWeights(p)
        s = p.stat(); os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns + 1_000_000))
        with self.assertRaisesRegex(ValueError, 'changed'):
            values.validate({'w': spec([2])})
        with self.assertRaisesRegex(ValueError, 'changed'): values['w']
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': '../a.safetensors'}}))
        with self.assertRaisesRegex(ValueError, 'relative'):
            SafeTensorWeights(self.root)
