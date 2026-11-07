import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from fxbex.native_fxb import build_native, validate_native, NativeFXBError, PRODUCER_MEMBER


class NativeFXBTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        (self.source / 'edfs').mkdir(parents=True)
        # Structural tests only: this is intentionally not an executable EDF.
        (self.source / 'edfs/k.edf').write_bytes(b'not-an-edf')
        self.manifest = {'format_version': 2, 'uuid': 'existing-native-id',
                         'created_at': '2026-11-07T00:00:00Z', 'versions': {},
                         'model': {'architecture': 'TestFixture'}, 'parallel_config': {},
                         'kernels': {'test': [{'bucket': {'tokenwise': {'input_size': 1}},
                                               'file': 'edfs/k.edf'}]}}
        self.save()
        self.output = self.root / 'out.fxb'

    def save(self):
        (self.source / 'manifest.json').write_text(json.dumps(self.manifest, indent=2))

    def build(self, **kwargs):
        return build_native(self.source, self.output, verify_official=False, **kwargs)

    def test_preserves_manifest_and_excludes_weights(self):
        (self.source / 'model.safetensors').write_bytes(b'weights')
        report = self.build(producer_metadata={'source_revision': 'real-source-id'})
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(archive.read('manifest.json'), (self.source / 'manifest.json').read_bytes())
            self.assertEqual(set(archive.namelist()), {'manifest.json', 'edfs/k.edf', PRODUCER_MEMBER})
            self.assertEqual(json.loads(archive.read(PRODUCER_MEMBER))['producer'], 'fxbex')
        self.assertTrue(report['weightless'])
        self.assertEqual(report['gates']['official_inspect'], 'not_run')
        self.assertEqual(report['gates']['npu_execution'], 'not_run')

    def test_official_reader_required_by_default(self):
        with patch('fxbex.native_fxb._official', side_effect=NativeFXBError('reader rejected fake EDF')):
            with self.assertRaisesRegex(NativeFXBError, 'fake EDF'):
                build_native(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.native-fxb-*')))

    def test_reader_pass_does_not_claim_runtime(self):
        with patch('fxbex.native_fxb._official') as official:
            report = build_native(self.source, self.output)
        official.assert_called_once()
        self.assertEqual(report['gates']['official_show'], 'passed')
        self.assertEqual(report['gates']['internal_metadata_reader'], 'passed')
        self.assertEqual(report['gates']['runtime_load'], 'not_run')
        self.assertEqual(report['gates']['model_compatibility'], 'not_run')

    def test_never_overwrites(self):
        self.output.write_bytes(b'existing')
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(self.output.read_bytes(), b'existing')

    def test_fxbex_manifest_refused(self):
        self.manifest = {'format': 'FXB_EX', 'runtime': {'adapter': 'runnable_dag'}}
        self.save()
        with self.assertRaisesRegex(NativeFXBError, 'EXPORT_BRIDGE_GAP'):
            self.build()
        self.assertFalse(self.output.exists())

    def test_non_native_references_refused(self):
        for name in ('edfs/k.runnable', 'weights/model.safetensors', '../k.edf', '/k.edf', 'edfs//k.edf'):
            with self.subTest(name=name):
                self.manifest['kernels']['test'][0]['file'] = name
                self.save()
                with self.assertRaises(NativeFXBError):
                    self.build()
        self.assertFalse(self.output.exists())

    def test_missing_and_symlink_refused(self):
        edf = self.source / 'edfs/k.edf'
        edf.unlink()
        with self.assertRaisesRegex(NativeFXBError, 'Missing regular'):
            self.build()
        external = self.root / 'external.edf'
        external.write_bytes(b'content')
        edf.symlink_to(external)
        with self.assertRaisesRegex(NativeFXBError, 'symlink'):
            self.build()

    def test_extra_archive_member_refused(self):
        self.build()
        with zipfile.ZipFile(self.output, 'a') as archive:
            archive.writestr('weights.bin', b'not weightless')
        with self.assertRaisesRegex(NativeFXBError, 'do not match'):
            validate_native(self.output, verify_official=False)

    def test_original_archive_without_extension_accepted(self):
        with zipfile.ZipFile(self.output, 'w') as archive:
            archive.write(self.source / 'manifest.json', 'manifest.json')
            archive.write(self.source / 'edfs/k.edf', 'edfs/k.edf')
        report = validate_native(self.output, verify_official=False)
        self.assertIsNone(report['producer_metadata'])
        self.assertEqual(report['gates']['internal_metadata_reader'], 'not_applicable')


if __name__ == '__main__':
    unittest.main()
