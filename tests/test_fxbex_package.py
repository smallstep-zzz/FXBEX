import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from fxbex.package import PackageError, build, extract, inspect, validate


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.write('manifest.json', {'model': {'name': 'test'}, 'compatibility': {},
                                     'runtime': {'adapter': 'test'}})
        self.write('execution_plan.json', {'steps': []})
        self.write('weights_index.json', {'weights': [{'path': 'weights/data.bin'}]})
        self.write('memory_plan.json', {})
        self.write('runtime_config.json', {})
        (self.source / 'weights').mkdir()
        (self.source / 'weights/data.bin').write_bytes(b'weights\x00')
        (self.source / 'kernel.edf').write_bytes(b'not-an-actual-edf')
        self.output = self.root / 'model.fxbex'

    def write(self, name, value):
        (self.source / name).write_text(json.dumps(value))

    def mutate(self, transform):
        build(self.source, self.output)
        with zipfile.ZipFile(self.output) as archive:
            members = [(i, archive.read(i)) for i in archive.infolist()]
        with zipfile.ZipFile(self.output, 'w') as archive:
            for info, data in transform(members):
                archive.writestr(info, data)

    def test_original_source_is_stored(self):
        build(self.source,self.output,standalone=True,source_url='stabilityai/TripoSR',source_revision='a'*40)
        self.assertEqual(inspect(self.output)['provenance'],{'source_url':'https://huggingface.co/stabilityai/TripoSR','revision':'a'*40,'status':'recorded'})

    def test_roundtrip_and_cleanup(self):
        build(self.source, self.output, standalone=True)
        manifest = inspect(self.output)
        self.assertTrue(manifest['standalone'])
        self.assertEqual(manifest['format'], 'FXB_EX')
        with extract(self.output) as directory:
            self.assertEqual((directory / 'weights/data.bin').read_bytes(), b'weights\x00')
        self.assertFalse(directory.exists())
        with self.assertRaises(FileExistsError):
            build(self.source, self.output)

    def test_corruption(self):
        self.mutate(lambda members: [(i, b'x' * len(data) if i.filename == 'kernel.edf' else data)
                                     for i, data in members])
        with self.assertRaisesRegex(PackageError, 'Checksum mismatch'):
            validate(self.output)
        exposed = False
        with self.assertRaisesRegex(PackageError, 'Checksum mismatch'):
            with extract(self.output):
                exposed = True
        self.assertFalse(exposed, 'corrupted contents must never be yielded')

    def test_extract_reads_large_member_once(self):
        build(self.source, self.output)
        calls = []
        original = zipfile.ZipFile.open

        def record(archive, name, *args, **kwargs):
            calls.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
            return original(archive, name, *args, **kwargs)

        with patch.object(zipfile.ZipFile, 'open', record):
            with extract(self.output):
                pass
        self.assertEqual(calls.count('weights/data.bin'), 1)

    def test_missing_required(self):
        (self.source / 'memory_plan.json').unlink()
        with self.assertRaises(PackageError):
            build(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_unknown_member(self):
        self.mutate(lambda members: members + [('unexpected', b'')])
        with self.assertRaisesRegex(PackageError, 'do not match'):
            validate(self.output)

    def test_unsafe_paths(self):
        for name in ('../escape', '/absolute', 'a\\b', 'a//b', './x', 'C:/x'):
            with self.subTest(name=name):
                self.output.unlink(missing_ok=True)
                self.mutate(lambda members: members + [(name, b'')])
                with self.assertRaisesRegex(PackageError, 'Unsafe'):
                    validate(self.output)

    def test_duplicate_and_symlink(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            self.mutate(lambda members: members + [members[0]])
        with self.assertRaisesRegex(PackageError, 'Duplicate'):
            validate(self.output)
        self.output.unlink()
        info = zipfile.ZipInfo('link')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.mutate(lambda members: members + [(info, b'/etc/passwd')])
        with self.assertRaisesRegex(PackageError, 'unsupported'):
            validate(self.output)

    def test_size_limits(self):
        build(self.source, self.output)
        with self.assertRaisesRegex(PackageError, 'size limit'):
            validate(self.output, max_total_size=10)
        with self.assertRaisesRegex(PackageError, 'size limit'):
            validate(self.output, max_json_size=10)

    def test_external_weights(self):
        self.write('weights_index.json', {'weights': [{'external_path': '/unavailable/weights'}]})
        build(self.source, self.output)
        self.output.unlink()
        with self.assertRaisesRegex(PackageError, 'Missing external'):
            build(self.source, self.output, standalone=True)
        self.assertFalse(self.output.exists())

    def test_standalone_copies_external_without_mutating_source(self):
        external = self.root / 'blob'
        external.write_bytes(b'external weights\x00')
        link = self.root / 'model.safetensors'
        link.symlink_to(external)
        index = {'weights': [{'external_path': str(link),
                              'sha256': hashlib.sha256(external.read_bytes()).hexdigest()}]}
        self.write('weights_index.json', index)
        build(self.source, self.output, standalone=True)
        with extract(self.output) as directory:
            rewritten = json.loads((directory / 'weights_index.json').read_text())
            self.assertNotIn('external_path', rewritten['weights'][0])
            self.assertEqual(rewritten['weights'][0]['path'], 'weights/model.safetensors')
            self.assertEqual((directory / 'weights/model.safetensors').read_bytes(), external.read_bytes())
        self.assertEqual(json.loads((self.source / 'weights_index.json').read_text()), index)
        self.assertFalse((self.source / 'weights/model.safetensors').exists())

    def test_standalone_collision_checksum_and_bad_name(self):
        external = self.root / 'external'
        external.write_bytes(b'weights')
        for extra, message in (({'name': 'data.bin'}, 'collision'),
                               ({'sha256': '0' * 64}, 'checksum mismatch'),
                               ({'name': '../escape'}, 'Unsafe'),
                               ({'name': 'nested/file'}, 'basename')):
            with self.subTest(extra=extra):
                self.write('weights_index.json', {'weights': [{'external_path': str(external), **extra}]})
                with self.assertRaisesRegex(PackageError, message):
                    build(self.source, self.output, standalone=True)
                self.assertFalse(self.output.exists())

    def test_standalone_external_size_limit(self):
        external = self.root / 'external'
        external.write_bytes(b'x' * 4096)
        self.write('weights_index.json', {'weights': [{'external_path': str(external)}]})
        with self.assertRaisesRegex(PackageError, 'size limit'):
            build(self.source, self.output, standalone=True, max_total_size=2048)
        self.assertFalse(self.output.exists())

    def test_missing_internal_weight(self):
        self.write('weights_index.json', {'weights': [{'path': 'weights/missing'}]})
        with self.assertRaisesRegex(PackageError, 'Missing packaged weight'):
            build(self.source, self.output)

    def test_output_in_source_and_source_link(self):
        with self.assertRaisesRegex(PackageError, 'outside'):
            build(self.source, self.source / 'nested/model.fxbex')
        (self.source / 'linked').symlink_to(self.source / 'kernel.edf')
        with self.assertRaisesRegex(PackageError, 'symlinks'):
            build(self.source, self.output)


if __name__ == '__main__':
    unittest.main()
