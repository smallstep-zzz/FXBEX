import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fxbex.cli import _save_result


class ImageOutputTests(unittest.TestCase):
    def test_png_destination_contains_image_and_report_stays_json(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = Path(__file__).resolve().parents[1] / 'examples/web/color-shapes.png'
            output = root / 'image.png'
            result = {'image_path': str(source)}
            self.assertEqual(_save_result(result, root, output), output)
            self.assertEqual(output.read_bytes(), source.read_bytes())
            self.assertEqual(json.loads((root / 'result.json').read_text())['image_path'], str(output))

    def test_non_image_cannot_be_written_as_png(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'requires a model'):
                _save_result({'text': 'hello'}, root, root / 'image.png')
            self.assertFalse((root / 'image.png').exists())

    def test_explicit_json_output_preserves_legacy_result(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / 'other.json'
            self.assertIsNone(_save_result({'image_path': '/some/image.png'}, root, output))
            self.assertEqual(json.loads(output.read_text())['image_path'], '/some/image.png')
