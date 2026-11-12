"""Model-to-FXB orchestration; actual native contracts, never guessed fingerprints."""
import json
import os
from pathlib import Path
import tempfile
import time


def build_program(entries,output,*,source_metadata=None,artifact_root=None,concurrency=4):
    """Delegate named calls and shared-state contracts to the semantic owner."""
    if type(concurrency) is not int or not 1<=concurrency<=32:raise ValueError('concurrency must be 1..32')
    from tplmachine.execution_program import export_program
    return export_program(entries,output,source_metadata=source_metadata,artifact_root=artifact_root,compile_workers=min(concurrency,8))


def resume_graph_build(work_dir, output, *, contract_provider=None):
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
                source_metadata=None, concurrency=4, work_dir=None,
                state_inputs=None,state_updates=None,user_output_indices=None,verification=None,**options):
    """Build a registered model, or package TCLMachine's static execution plan.

    A HF ID/local asset directory uses the official architecture-aware compiler.
    A torch module delegates capture, partitioning and EDF compilation to TCLMachine.
    No model weights are placed into the ZIP. Hardware acceptance remains separate.
    """
    from .native_fxb import NativeFXBError, build_native
    output=Path(output).absolute()
    custom_plan = not isinstance(model, (str, os.PathLike)) and contract_provider is None
    if output.suffix != '.fxb' and not (custom_plan and output.suffix == '.fxbx'):
        raise ValueError('Official output uses .fxb; custom execution plans may use .fxbx')
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    if type(concurrency) is not int or not 1<=concurrency<=32:
        raise ValueError('concurrency must be 1..32')
    allowed={'tensor_parallel_size','pipeline_parallel_size','max_model_len','optim_level',
             'compiler_cache_dir','no_compiler_cache','tokenwise_buckets','attention_buckets',
             'build_report','dump_tc'}
    if set(options)-allowed:raise ValueError('unsupported compiler options: '+','.join(sorted(set(options)-allowed)))
    if verification is not None:
        required={'validation_inputs','atol','rtol'}
        if (not isinstance(verification,dict) or not required <= set(verification)
                or set(verification)-required-{'repeats','replay_timeout'}):
            raise ValueError('verification requires validation_inputs/atol/rtol and optional repeats/replay_timeout')
        if isinstance(model,(str,os.PathLike)) or contract_provider is not None:
            raise ValueError('Producer-owned verification requires the TCLMachine Torch route')
    output.parent.mkdir(parents=True,exist_ok=True)
    work=Path(work_dir).absolute() if work_dir else output.parent/(output.name+'.build')
    work.mkdir(parents=True,exist_ok=False)
    report={'status':'building','output':str(output),'weights_included':False,'stages':[],
            'npu_execution':'not_run','source_metadata':dict(source_metadata or {})}
    started=time.monotonic()
    try:
        if isinstance(model,(str,os.PathLike)):
            if state_inputs or state_updates or user_output_indices is not None:raise ValueError('state/output annotations require a Torch graph')
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
            if options:
                raise ValueError('STATIC_GRAPH_OPTIONS_UNSUPPORTED: custom graph export cannot honor model context/bucket/compiler options: '+','.join(sorted(options)))
            if example_inputs is None:raise ValueError('torch models require example_inputs')
            if contract_provider is None and verification is not None:
                from tclmachine import build_verified_model
                result=build_verified_model(model,tuple(example_inputs),output,source_metadata=source_metadata,
                    state_inputs=state_inputs,state_updates=state_updates,user_output_indices=user_output_indices,
                    compile_workers=min(concurrency,8),**verification)
                report.update(route='tclmachine_verified_build',status=result['status'],result=result,
                              npu_execution='see_producer_replay_evidence')
                return report
            if contract_provider is None:
                from tplmachine.execution_plan import export_model
                report['route']='tclmachine_execution_plan'
                result=export_model(model,tuple(example_inputs),output,source_metadata=source_metadata,
                    state_inputs=state_inputs,state_updates=state_updates,user_output_indices=user_output_indices,compile_workers=min(concurrency,8))
                report.update(status='built_not_runtime_validated',result=result)
                return report
            if state_inputs or state_updates or user_output_indices is not None:raise ValueError('legacy single-EDF contract has no state/output annotations')
            from tplmachine.edf_export import export_edf
            from .graph_fxb import export_graph
            import torch
            from tplmachine.graph_capture import capture_for_export
            inputs=tuple(example_inputs)
            exported=capture_for_export(model,inputs)
            edf=dict(export_edf(exported,inputs).contract)
            if not isinstance(model,torch.export.ExportedProgram):
                report['source_metadata'].setdefault('architecture',type(model).__name__)
                report['source_metadata'].setdefault('python_class',type(model).__module__+'.'+type(model).__qualname__)
            (work/'edf-contract.json').write_text(json.dumps(edf,indent=2)+'\n')
            report['artifacts']={'edf':edf['edf_path'],'edf_contract':str(work/'edf-contract.json')}
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
