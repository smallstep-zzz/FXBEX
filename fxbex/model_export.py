"""Model-to-FXB orchestration; actual native contracts, never guessed fingerprints."""
import json
import os
from pathlib import Path
import tempfile
import time


def resume_graph_build(work_dir, output, *, contract_provider):
    """Resume native packaging without repeating a successful EDF compilation."""
    from .graph_fxb import export_graph
    work=Path(work_dir)
    report=json.loads((work/'result.json').read_text())
    edf=json.loads((work/'edf-contract.json').read_text())
    return export_graph(edf,output,model_metadata=report['source_metadata'],
                        contract_provider=contract_provider)


def _official_build(model, output, *, concurrency, options):
    from furiosa.llm_native.build import build_fxb
    build_fxb(str(model), str(output), concurrency=concurrency, **options)


def build_model(model, output, *, example_inputs=None, contract_provider=None,
                source_metadata=None, concurrency=4, work_dir=None, **options):
    """Build a registered model, or export a static graph with a genuine contract provider.

    A HF ID/local asset directory uses the official architecture-aware compiler.
    A torch module uses TCLMachine's lifted-input EDF export and FXBEX graph plan.
    No model weights are placed into the ZIP. Hardware acceptance remains separate.
    """
    from .native_fxb import NativeFXBError, build_native
    output=Path(output).absolute()
    if output.suffix!='.fxb':raise ValueError('output must end in .fxb')
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    if type(concurrency) is not int or not 1<=concurrency<=32:
        raise ValueError('concurrency must be 1..32')
    allowed={'tensor_parallel_size','pipeline_parallel_size','max_model_len','optim_level',
             'compiler_cache_dir','no_compiler_cache','tokenwise_buckets','attention_buckets',
             'build_report','dump_tc'}
    if set(options)-allowed:raise ValueError('unsupported compiler options: '+','.join(sorted(set(options)-allowed)))
    output.parent.mkdir(parents=True,exist_ok=True)
    work=Path(work_dir).absolute() if work_dir else output.parent/(output.name+'.build')
    work.mkdir(parents=True,exist_ok=False)
    report={'status':'building','output':str(output),'weights_included':False,'stages':[],
            'npu_execution':'not_run','source_metadata':dict(source_metadata or {})}
    started=time.monotonic()
    try:
        if isinstance(model,(str,os.PathLike)):
            report['route']='official_registered_model'
            native=work/'official.fxb'
            _official_build(model,native,concurrency=concurrency,options=options)
            if not native.is_file():raise NativeFXBError('official builder returned without an FXB')
            report['stages'].append({'name':'official_compile','status':'passed'})
            # The official artifact is the source of its model fingerprint and ABI.
            import zipfile,shutil
            from .native_fxb import validate_native, _name
            validate_native(native,verify_official=True)
            extracted=work/'native';extracted.mkdir()
            with zipfile.ZipFile(native) as archive:
                for entry in archive.infolist():
                    name=_name(entry.filename)
                    if name=='fxbex.provenance.json':continue
                    dest=extracted/name;dest.parent.mkdir(parents=True,exist_ok=True)
                    with archive.open(entry) as src,dest.open('xb') as dst:
                        shutil.copyfileobj(src,dst,1024**2)
            result=build_native(extracted,output,producer_metadata={
                'operation_source':'official_model_build','source_model':str(model),
                'source_metadata':report['source_metadata']})
        else:
            report['route']='custom_static_graph'
            if example_inputs is None:raise ValueError('torch models require example_inputs')
            from tplmachine.edf_export import export_edf
            from .graph_fxb import export_graph
            import torch
            inputs=tuple(example_inputs)
            exported=model if isinstance(model,torch.export.ExportedProgram) else torch.export.export(model,inputs,strict=False)
            edf=export_edf(exported,inputs).write(work/'model.edf')
            if not isinstance(model,torch.export.ExportedProgram):
                report['source_metadata'].setdefault('architecture',type(model).__name__)
                report['source_metadata'].setdefault('python_class',type(model).__module__+'.'+type(model).__qualname__)
            (work/'edf-contract.json').write_text(json.dumps(edf,indent=2)+'\n')
            report['artifacts']={'edf':str(work/'model.edf'),'edf_contract':str(work/'edf-contract.json')}
            report['stages'].append({'name':'graph_edf_compile','status':'passed'})
            result=export_graph(edf,output,contract_provider=contract_provider,
                                model_metadata=report['source_metadata'])
        report.update(status='built_not_runtime_validated',result=result)
        return report
    except Exception as exc:
        report.update(status='blocked',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        report['seconds']=time.monotonic()-started
        (work/'result.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
