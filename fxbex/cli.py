"""CLI for local FXB_EX packages; SDK is imported only for runtime operations."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
from .package import build, inspect
from .validation import validate


def _runtime_call(callback, log_path, verbose=False):
    """Keep framework diagnostics separate from answers and JSON results."""
    if verbose:return callback()
    import contextlib
    with log_path.open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        return callback()


def _load_with_progress(callback, stream, interval=5):
    """Report elapsed time without inventing a completion percentage."""
    stopped = threading.Event()
    started = time.monotonic()
    interactive = stream.isatty()
    displayed = False
    def report():
        nonlocal displayed
        while not stopped.wait(interval):
            message = f'Model loading... {int(time.monotonic() - started)}s elapsed'
            print(('\r' if interactive else '') + message, end='' if interactive else '\n', file=stream, flush=True)
            displayed = True
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        return callback()
    finally:
        stopped.set()
        thread.join()
        if interactive and displayed:
            print(file=stream, flush=True)


def _main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    commands={'build','convert','inspect','validate','describe','sample','doctor','ui','run','chat','predict','serve','fetch','cache'}
    if argv and not argv[0].startswith('-') and argv[0] not in commands:
        argv.insert(0,'convert')
    parser = argparse.ArgumentParser(
        prog='fxbx', description='모델을 NPU FXB로 변환하고 컴파일 산출물을 패키징합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  fxbx 모델 -o 결과.fxb
  fxbx inspect 결과.fxb
  fxbx build ./compiled_model -o model.fxbex
  fxbx inspect model.fxbex
  fxbx validate model.fxbex --integrity-only
  fxbx validate model.fxbex --weights-dir ./weights
  fxbx run model.fxbex --input inputs.json -o result.json
  fxbx chat model.fxbex
  fxbx serve model.fxbex --port 8000
  fxbx doctor

Validation checks static readiness; it does not run inference or prove NPU execution.
Model-specific exporters and capture commands belong to their adapter plugins.""")
    sub = parser.add_subparsers(dest='action', required=True)
    from . import __version__
    parser.add_argument('--version',action='version',version='%(prog)s '+__version__)
    p = sub.add_parser('convert', help='모델을 NPU용 FXB로 변환')
    p.add_argument('model', help='Hugging Face 모델 ID 또는 로컬 모델 디렉터리')
    p.add_argument('-o','--output',required=True,help='새 .fxb 파일')
    p.add_argument('--concurrency',type=int,default=4)
    p = sub.add_parser('build', help='Pack a compiled directory with checksums (no compilation)')
    p.add_argument('source', help='Compiled directory containing manifest and required JSON configurations')
    p.add_argument('-o', '--output', required=True, help='New .fxbex file; existing files are never overwritten')
    p.add_argument('--source-model',help='Original HF model ID or HTTPS model URL')
    p.add_argument('--source-revision',help='Original checkpoint revision (prefer immutable commit)')
    p.add_argument('--standalone', action='store_true', help='Copy referenced external weights into the archive')
    p = sub.add_parser('inspect', help='Verify integrity and display the manifest; no SDK needed')
    p.add_argument('package', help='Path to a .fxb or .fxbex archive')
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
            if name == 'run':
                p.add_argument('--sample', action='store_true', help='Run the packaged example once, including for chat models')
                p.add_argument('--image', help='Use an image file instead of the packaged image sample')
        else:
            p.add_argument('--port', type=int, default=8000, help='HTTP port (default: 8000)')
    p=sub.add_parser('fetch', help='Download a catalog release into the shared FXBEX cache')
    p.add_argument('package')
    p.add_argument('--refresh', action='store_true', help='Recheck catalog and force a fresh HF download')
    p=sub.add_parser('cache', help='List or clear FXBEX-managed model downloads only')
    p.add_argument('operation', choices=['list','clear'])
    p.add_argument('--yes', action='store_true', help='Confirm deletion of managed model downloads')
    args=parser.parse_args(argv)
    if args.action=='cache':
        from .settings import cache_root
        root=cache_root()/'models'
        files=sorted(root.rglob('*.fxbex')) if root.exists() else []
        if args.operation=='clear':
            if not args.yes:raise ValueError('Use cache clear --yes to delete FXBEX-managed downloads')
            import shutil
            if root.is_symlink():raise ValueError('Refusing to clear a symlinked cache root')
            if root.exists():shutil.rmtree(root)
        print(json.dumps({'cache':str(root),'files':[str(p) for p in files],'cleared':args.operation=='clear'}))
        return 0
    if args.action=='fetch':
        from .hub import resolve_package
        print(json.dumps({'package':str(resolve_package(args.package,refresh=args.refresh))}))
        return 0
    if args.action in ('convert','run','chat','predict','serve','doctor') or (args.action=='validate' and not args.integrity_only):
        from .settings import runtime_python
        interpreter=runtime_python()
        if Path(interpreter)!=Path(sys.executable):
            import subprocess
            print('Using SDK Python: '+interpreter,file=sys.stderr,flush=True)
            code=subprocess.call([interpreter,'-m','fxbx',*(sys.argv[1:] if argv is None else map(str,argv))])
            return code if code>=0 else 128-code
    if args.action=='convert':
        from .model_export import build_model
        result=build_model(args.model,args.output,concurrency=args.concurrency)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    if args.action=='inspect' and Path(args.package).is_file():
        from .native_fxb import MAX_JSON_SIZE
        import zipfile
        with zipfile.ZipFile(args.package) as archive:
            if archive.getinfo('manifest.json').file_size>MAX_JSON_SIZE:
                raise ValueError('Manifest exceeds JSON limit')
            manifest=json.loads(archive.read('manifest.json'))
        if isinstance(manifest,dict) and manifest.get('format_version')==2:
            from .native_fxb import validate_native
            result=validate_native(args.package,verify_official=False)
            producer=result.pop('producer_metadata') or {}
            result['model']=manifest['model']
            if 'execution_plan' in producer:
                from .execution_fxb import read_execution_plan,_memory_gate
                plan=read_execution_plan(args.package)
                result['execution_plan']=plan
                result['compatibility']=plan['compatibility']
                if args.memory:result['memory_plan']=_memory_gate(plan)
            elif 'graph_plan' in producer:
                from .graph_fxb import read_graph_plan
                plan=read_graph_plan(args.package)
                result['execution_plan']=plan
                result['compatibility']=plan['compatibility']
            else:
                result['execution_plan']=None
                result['compatibility']=manifest['versions']
                result['runtime']='furiosa_llm.LLM(model_dir, fxb=path)'
            result['compatibility_scope']='기록된 계약 확인. 현재 환경 적합성과 실행 성공은 미검증'
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0
    if args.action=='chat' and not args.input:
        print('Welcome to FXBEX Chat!\nModel: '+args.package+
              '\n모델을 준비하고 있습니다. 적재가 완료되면 >>> 뒤에 질문을 입력하세요.'+
              '\n/reset: 대화 초기화 · /quit: 종료 및 모델 해제\n',file=sys.stderr,flush=True)
    if args.action in ('run','chat','predict','serve','describe','sample','inspect','validate'):
        from .hub import resolve_package
        args.package=str(resolve_package(args.package))
    sample_contract=None
    if args.action=='run' and args.image and args.input:
        raise ValueError('Use --image or --input, not both')
    if args.action=='run' and not args.input:
        from .schema import describe
        sample_contract=describe(args.package)
        if args.image and (sample_contract['method']!='predict' or 'image' not in sample_contract['inputs']):
            raise ValueError('This package does not declare an image input; inspect its input contract')
        if args.image and not Path(args.image).is_file():
            raise ValueError('Image file does not exist')
        if sample_contract['method']=='chat' and not args.sample:
            sample_contract=None
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
        session=Session(args.package,report_dir=args.report_dir,weights_dir=args.weights_dir)
        log_path=session.report_dir/'runtime.log'
        print(f'Loading package; diagnostics: {log_path}',file=sys.stderr,flush=True)
        def invoke(callback):return _runtime_call(callback,log_path,args.verbose)
        progress_stream = sys.stderr
        _load_with_progress(lambda: invoke(session.__enter__), progress_stream)
        try:
            for message in package_compatibility(session.manifest):
                print('Warning: '+message,file=sys.stderr,flush=True)
            print(f'NPU package ready; reports: {session.report_dir}',file=sys.stderr,flush=True)
            from .execution import memory_text
            print('Memory: '+memory_text(getattr(session,'memory',None)),file=sys.stderr,flush=True)
            if args.action=='serve':
                from .server import serve
                serve(session,args.port,invoke=invoke);return
            if sample_contract is not None:
                from .schema import resolve_sample
                data=resolve_sample(sample_contract,session.root,session.manifest)
                if args.image:
                    data['image']=str(Path(args.image).resolve())
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
        print(f'fxbx: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
