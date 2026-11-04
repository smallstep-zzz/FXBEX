"""Optional declarative I/O help. Reading examples never executes package code."""
import json
from .package import inspect,PackageError

def describe(package):
    manifest=inspect(package)
    contract=manifest.get('io_contract')
    if contract is None:raise PackageError('No io_contract: ask the publisher to provide input/output examples.')
    if not isinstance(contract,dict) or contract.get('schema_version')!=1:
        raise PackageError('Unsupported io_contract')
    if contract.get('method') not in ('chat','predict') or not isinstance(contract.get('example_input'),dict):
        raise PackageError('io_contract requires method and named example_input')
    for key in ('inputs','outputs'):
        if not isinstance(contract.get(key),dict):raise PackageError('io_contract requires '+key)
    if len(json.dumps(contract).encode())>65536:raise PackageError('I/O help exceeds 64KiB')
    return contract

def example_inputs(package):
    return describe(package)['example_input']

def resolve_sample(contract,root,manifest):
    """Resolve explicit packaged assets only; plain strings remain ordinary text."""
    from pathlib import Path
    root=Path(root).resolve()
    def visit(value):
        if isinstance(value,dict):
            if set(value)=={'$asset'}:
                name=value['$asset']
                if not isinstance(name,str) or name not in manifest['files']:raise PackageError('Sample asset is not in integrity map')
                path=(root/name).resolve()
                if not path.is_relative_to(root) or not path.is_file():raise PackageError('Invalid sample asset')
                return str(path)
            return {k:visit(v) for k,v in value.items()}
        if isinstance(value,list):return [visit(v) for v in value]
        return value
    return visit(contract['example_input'])
