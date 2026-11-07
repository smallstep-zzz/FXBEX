"""Explicit SDK integration check; never collected as a unit/NPU test."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import zipfile

from fxbex.native_fxb import build_native, _contract, MAX_JSON_SIZE


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Genuine existing native FXB')
    parser.add_argument('--work-dir', type=Path, required=True, help='New evidence directory')
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    extracted = args.work_dir / 'native-source'
    extracted.mkdir()
    with zipfile.ZipFile(args.source) as archive:
        if archive.getinfo('manifest.json').file_size > MAX_JSON_SIZE:
            raise ValueError('Oversized manifest')
        original = archive.read('manifest.json')
        files, _ = _contract(json.loads(original))
        (extracted / 'manifest.json').write_bytes(original)
        original_members = {}
        for name in sorted(files):
            target = extracted / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination, length=1024**2)
            member = archive.getinfo(name)
            original_members[name] = (member.file_size, member.CRC)
    output = args.work_dir / 'qwen3-8b-fp8-fxbex-producer.fxb'
    source_sha = digest(args.source)
    report = build_native(extracted, output, producer_metadata={
        'source_native_artifact': args.source.name,
        'source_native_sha256': source_sha,
    })
    with zipfile.ZipFile(output) as archive:
        report['manifest_bytes_preserved'] = archive.read('manifest.json') == original
        report['edf_sizes_and_crcs_preserved'] = all(
            (archive.getinfo(name).file_size, archive.getinfo(name).CRC) == properties
            for name, properties in original_members.items())
    assert report['manifest_bytes_preserved'] and report['edf_sizes_and_crcs_preserved']
    from furiosa.llm_native import fxb, build
    for name in ('show', 'inspect'):
        rendered = getattr(fxb, name)(str(output))
        (args.work_dir / f'official-{name}.txt').write_text(str(rendered))
    report.update({'source_native_sha256': source_sha, 'output_sha256': digest(output),
                   'output_size_bytes': output.stat().st_size,
                   'public_build_api': {name: str(inspect.signature(getattr(build, name)))
                                        for name in dir(build) if not name.startswith('_')},
                   'public_fxb_api': [name for name in dir(fxb) if not name.startswith('_')],
                   'extension_status': 'experimental_reader_accepted_runtime_not_run'})
    (args.work_dir / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
