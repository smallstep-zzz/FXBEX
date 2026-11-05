"""CLI for local FXB_EX packages; SDK is imported only for runtime operations."""
import argparse
import json
from pathlib import Path
import sys
from .package import build, inspect
from .validation import validate


def _runtime_call(callback, log_path, verbose=False):
    """Keep framework diagnostics separate from answers and JSON results."""
    if verbose:return callback()
    import contextlib
    with log_path.open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        return callback()


def _main(argv=None):
    parser = argparse.ArgumentParser(
        prog='fxbex', description='Build, inspect and run compiled FXB_EX model packages.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  fxbex build ./compiled_model -o model.fxbex
  fxbex inspect model.fxbex
  fxbex validate model.fxbex --integrity-only
  fxbex validate model.fxbex --weights-dir ./weights
  fxbex run model.fxbex --input inputs.json -o result.json
  fxbex chat model.fxbex
  fxbex serve model.fxbex --port 8000
  fxbex doctor

Validation checks static readiness; it does not run inference or prove NPU execution.
Model-specific exporters and capture commands belong to their adapter plugins.""")
    sub = parser.add_subparsers(dest='action', required=True)
    from . import __version__
    parser.add_argument('--version',action='version',version='%(prog)s '+__version__)
    p = sub.add_parser('build', help='Pack a compiled directory with checksums (no compilation)')
    p.add_argument('source', help='Compiled directory containing manifest and required JSON configurations')
    p.add_argument('-o', '--output', required=True, help='New .fxbex file; existing files are never overwritten')
    p.add_argument('--source-model',help='Original HF model ID or HTTPS model URL')
    p.add_argument('--source-revision',help='Original checkpoint revision (prefer immutable commit)')
    p.add_argument('--standalone', action='store_true', help='Copy referenced external weights into the archive')
    p = sub.add_parser('inspect', help='Verify integrity and display the manifest; no SDK needed')
    p.add_argument('package', help='Path to a .fxbex archive')
    p.add_argument('--memory', action='store_true', help='Show logical memory/lifetime plan without loading the NPU')
    p = sub.add_parser('validate', help='Check integrity, SDK, adapter, weights and device availability without inference')
    p.add_argument('package', help='Path to a .fxbex archive')
    p.add_argument('--integrity-only', action='store_true', help='Verify archive only; skip SDK, weights and device checks')
    p.add_argument('--weights-dir', help='Override the external checkpoint/weights directory')
    for name in ('describe','sample'):
        p=sub.add_parser(name,help='Show I/O contract or example input without running the model')
        p.add_argument('package')
    sub.add_parser('doctor', help='Report installed SDK versions and RNGD device status')
    p = sub.add_parser('ui', help='Launch the local package interface')
    p.add_argument('--port', type=int, default=7861, help='HTTP port (default: 7861)')
    for name, help_text in (
        ('run', 'Run JSON inputs or start interactive chat for a chat adapter'),
        ('chat', 'Chat interactively, or process a messages JSON file'),
        ('predict', 'Run a tensor DAG with named JSON inputs'),
        ('serve', 'Serve the loaded package over local HTTP'),
    ):
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument('package', help='Path to a .fxbex archive')
        p.add_argument('--verbose',action='store_true',help='Show internal SDK/compiler diagnostics on the terminal')
        p.add_argument('--report-dir', help='Directory for runtime logs and evidence')
        p.add_argument('--weights-dir', help='Override external checkpoint/weights directory')
        if name != 'serve':
            p.add_argument('--input', help='JSON file: named tensors for predict, or {"messages": [...]} for chat')
            p.add_argument('-o', '--output', help='Write response JSON to this file')
            p.add_argument('--max-new-tokens', type=int, default=32, help='Chat token limit (default: 32)')
        else:
            p.add_argument('--port', type=int, default=8000, help='HTTP port (default: 8000)')
    args=parser.parse_args(argv)
    if args.action in ('run','chat','predict','serve','doctor') or (args.action=='validate' and not args.integrity_only):
        from .settings import runtime_python
        interpreter=runtime_python()
        if Path(interpreter)!=Path(sys.executable):
            import subprocess
            print('Using SDK Python: '+interpreter,file=sys.stderr,flush=True)
            code=subprocess.call([interpreter,'-m','fxbex',*(sys.argv[1:] if argv is None else map(str,argv))])
            return code if code>=0 else 128-code
    if args.action=='chat' and not args.input:
        print('Welcome to FXBEX Chat!\nModel: '+args.package+
              '\n모델을 준비하고 있습니다. 적재가 완료되면 >>> 뒤에 질문을 입력하세요.'+
              '\n/reset: 대화 초기화 · /quit: 종료 및 모델 해제\n',file=sys.stderr,flush=True)
    if args.action in ('run','chat','predict','serve','describe','sample','inspect','validate'):
        from .hub import resolve_package
        args.package=str(resolve_package(args.package))
    sample_contract=None
    if args.action=='run' and not args.input:
        from .schema import describe
        sample_contract=describe(args.package)
    if args.action=='build':result={'package':str(build(args.source,args.output,standalone=args.standalone,source_url=args.source_model,source_revision=args.source_revision))}
    elif args.action == 'inspect':
        m=inspect(args.package)
        result={'model':m['model'],'runtime':m['runtime'],'compatibility':m['compatibility'],
                'standalone':m['standalone'],'bundled_python':bool(m['runtime'].get('bundled_python')),
                'delivery':m.get('delivery'), 'provenance':m.get('provenance',{'status':'unknown'}), 'files':len(m['files']),
                'artifact_bytes':sum(x['size'] for x in m['files'].values()),'integrity_verified':True}
        if args.memory:
            from .memory import inspect_archive, summarize
            result['memory_plan'] = summarize(inspect_archive(args.package))
    elif args.action == 'validate':
        result = validate(args.package, integrity_only=args.integrity_only, weights_dir=args.weights_dir)
    elif args.action in ('describe','sample'):
        from .schema import describe,example_inputs
        result=describe(args.package) if args.action=='describe' else example_inputs(args.package)
    elif args.action=='doctor':
        from .runtime import doctor
        result=doctor()
    elif args.action=='ui':
        from .ui import launch
        launch(args.port);return
    else:
        from .runtime import Session
        # SDK/compiler diagnostics may print; keep JSON results on stdout.
        import contextlib
        if args.action == 'predict' and not args.input:
            raise ValueError('--input JSON file is required for tensor prediction')
        from .runtime import package_compatibility
        for message in package_compatibility(inspect(args.package)):
            print('Warning: '+message,file=sys.stderr,flush=True)
        session=Session(args.package,report_dir=args.report_dir,weights_dir=args.weights_dir)
        log_path=session.report_dir/'runtime.log'
        print(f'Loading package; diagnostics: {log_path}',file=sys.stderr,flush=True)
        def invoke(callback):return _runtime_call(callback,log_path,args.verbose)
        invoke(session.__enter__)
        print(f'NPU package ready; reports: {session.report_dir}',file=sys.stderr,flush=True)
        from .execution import memory_text
        print('Memory: '+memory_text(getattr(session,'memory',None)),file=sys.stderr,flush=True)
        try:
            if args.action=='serve':
                from .server import serve
                serve(session,args.port);return
            if sample_contract is not None:
                from .schema import resolve_sample
                data=resolve_sample(sample_contract,session.root,session.manifest)
                if sample_contract['method']=='chat':
                    result=invoke(lambda: session.chat(data['messages'],data.get('max_new_tokens',args.max_new_tokens)))
                else:result=invoke(lambda: session.predict(data))
            elif args.action=='predict' or (args.action=='run' and session.manifest['runtime']['adapter']=='runnable_dag'):
                if not args.input:raise ValueError('--input JSON file is required for tensor prediction')
                result=invoke(lambda: session.predict(json.loads(Path(args.input).read_text())))
            elif args.input:
                data=json.loads(Path(args.input).read_text())
                result=invoke(lambda: session.chat(data['messages'],data.get('max_new_tokens',args.max_new_tokens)))
            else:
                messages=[]
                print('모델 적재 완료. 대화를 시작하세요. (/reset: 초기화 · /quit: 종료)',flush=True)
                while True:
                    try:prompt=input('>>> ')
                    except EOFError:break
                    if prompt.strip()=='/quit':break
                    if prompt.strip()=='/reset':messages=[];continue
                    if not prompt.strip():continue
                    candidate=messages+[{'role':'user','content':prompt}]
                    result=invoke(lambda: session.chat(candidate,args.max_new_tokens))
                    messages=candidate+[{'role':'assistant','content':result['text']}]
                    print(result['text'],flush=True)
                return
            if isinstance(result,dict):result.setdefault('memory',getattr(session,'memory',None))
        finally:invoke(lambda: session.__exit__(None,None,None))
        result['report_dir']=str(session.report_dir)
        result_path=Path(args.output).resolve() if args.output else session.report_dir/'result.json'
        result['result_file']=str(result_path)
        result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        if not args.verbose:
            # Full raw diagnostics remain in result.json and runtime.log.
            result={key:value for key,value in result.items() if key not in
                    ('memory','compiler_calls','replay_hits','npu_tasks','validated_audit','current_request_traced')}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result.get('valid', True) else 1

def main(argv=None):
    try:
        return _main(argv) or 0
    except Exception as exc:
        print(f'fxbex: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
