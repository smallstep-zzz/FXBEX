"""Native v2 envelope for an explicit fxbX static-graph consumer.

This is not a Furiosa-LLM architecture registration or compatibility assertion.
Only metadata is assembled here; official reader and execution gates run later.
"""
from collections.abc import Mapping
from datetime import datetime, timezone
import re
from uuid import uuid4


class StaticGraphContractProvider:
    """Bind one stateless EDF to its graph, ABI and recorded compiler identity."""

    def provide(self, *, model_metadata, edf_export):
        # Lazy import keeps the graph writer/provider dependency acyclic.
        from .graph_fxb import GraphFXBError, _bindings, _bytes, _sha
        from .native_fxb import MAX_JSON_SIZE

        def reject(message):
            raise GraphFXBError('STATIC_GRAPH_CONTRACT: ' + message)

        def text(value, label, maximum=256):
            if (not isinstance(value, str) or not value.strip()
                    or len(value) > maximum or any(ord(c) < 32 for c in value)):
                reject(label + ' must be bounded nonempty text')
            return value

        def digest(value, label):
            if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
                reject(label + ' must be a lowercase SHA256')
            return value

        if not isinstance(model_metadata, Mapping) or not isinstance(edf_export, Mapping):
            reject('model metadata and EDF export must be mappings')
        if len(_bytes([model_metadata, edf_export])) > MAX_JSON_SIZE:
            reject('metadata exceeds native JSON limit')
        architecture = text(model_metadata.get('architecture'), 'actual architecture')
        source = edf_export.get('source_identity')
        if not isinstance(source, Mapping):
            reject('recorded source_identity is required')
        digest(source.get('graph_sha256'), 'source graph hash')
        if 'graph_code_sha256' in source:
            digest(source['graph_code_sha256'], 'source graph-code hash')
        digest(edf_export.get('edf_sha256'), 'EDF hash')
        if (type(edf_export.get('expected_num_fusion')) is not int
                or edf_export['expected_num_fusion'] != 8):
            reject('only an EDF recorded for fusion8 is supported')
        if edf_export.get('target_npu') not in ('renegade', 'renegade-8pe'):
            reject('a recorded Renegade target is required')
        inputs, outputs, abi = (edf_export.get(key) for key in ('inputs', 'outputs', 'abi'))
        if (not isinstance(inputs, list) or not isinstance(outputs, list)
                or not 1 <= len(inputs) <= 4096 or not 1 <= len(outputs) <= 4096):
            reject('bounded nonempty tensor input/output lists are required')
        _bindings(inputs, outputs, abi)
        for spec in inputs + outputs:
            if (len(spec['shape']) > 16 or any(n <= 0 or n > 2**63 - 1 for n in spec['shape'])
                    or any(n > 2**63 - 1 for n in spec['stride'])):
                reject('positive static shapes and bounded strides are required')
            text(spec['dtype'], 'tensor dtype', 64)
        if any(len(value) > 65536 for value in abi['inputs'] + abi['outputs']):
            reject('ABI description exceeds size limit')
        user = sorted((spec for spec in inputs if spec['kind'] == 'user_input'),
                      key=lambda spec: spec['user_index'])
        if not user or not user[0]['shape']:
            reject('first user tensor needs a static leading dimension for the bucket')
        compatibility = edf_export.get('compatibility')
        if not isinstance(compatibility, Mapping):
            reject('recorded SDK package versions are required')
        packages = compatibility.get('packages', compatibility)
        if not isinstance(packages, Mapping):
            reject('SDK package versions must be a mapping')
        for name in ('torch', 'furiosa-torch', 'furiosa-tcc'):
            value = text(packages.get(name), name + ' version', 96)
            if value.lower() in ('unknown', 'none', 'unavailable'):
                reject(name + ' version is unknown')
        compiler = edf_export.get('compiler_identity')
        if not isinstance(compiler, Mapping):
            reject('recorded compiler_identity version/revision is required')
        compiler = {key: text(compiler.get(key), 'compiler ' + key, 128)
                    for key in ('version', 'revision')}
        if any(value.lower() in ('unknown', 'none', 'unavailable') for value in compiler.values()):
            reject('compiler version/revision is unknown')
        if 'compiler_config' not in edf_export:
            reject('recorded compiler_config is required, including an explicit None default')
        config = edf_export['compiler_config']
        if config is not None and not isinstance(config, Mapping):
            reject('compiler_config must be its recorded mapping or None')

        device = {'fusion': 8, 'mesh': 'single'}
        bucket = {'tokenwise': {'input_size': user[0]['shape'][0]}}
        fingerprint = _sha({
            'schema': 'fxbx.static-graph-fingerprint.v1',
            'model_metadata': model_metadata, 'source_identity': source,
            'inputs': inputs, 'outputs': outputs, 'abi': abi,
            'edf_sha256': edf_export['edf_sha256'],
            'compatibility': compatibility, 'compiler_identity': compiler,
            'compiler_config': config, 'target_npu': edf_export['target_npu'],
            'device_config': device,
        })
        manifest = {
            'format_version': 2, 'uuid': str(uuid4()),
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
            'versions': {'furiosa_compiler': compiler},
            'model': {'architecture': architecture, 'fxbx_graph_fingerprint': fingerprint},
            'parallel_config': {'tensor_parallel_size': 8, 'pipeline_parallel_size': 1},
            'kernels': {'forward': [{'bucket': bucket, 'file': 'edfs/forward.edf'}]},
        }
        return {
            'manifest': manifest, 'kernel_name': 'forward', 'bucket': bucket,
            'member': 'edfs/forward.edf', 'model_metadata_sha256': _sha(model_metadata),
            'device_config': device,
            'source': {'kind': 'fxbx.static-graph', 'schema': 1,
                       'fxbx_graph_fingerprint': fingerprint,
                       'explicit_consumer_required': True, 'official_llm_dispatch': False},
        }
