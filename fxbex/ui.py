"""Reusable Gradio shell. Agents supply model-specific controls through controls(run)."""
import json
from pathlib import Path
import shutil
import threading
import time


def save_package(source, directory):
    """Copy a local package without silently replacing an existing delivery."""
    source=Path(source).resolve();directory=Path(directory).resolve()
    if source.suffix != '.fxbex' or not source.is_file():raise ValueError('Expected a .fxbex file')
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/source.name
    if source==target:return str(target)
    with source.open('rb') as src, target.open('xb') as dst:
        try:shutil.copyfileobj(src,dst)
        except BaseException:
            target.unlink(missing_ok=True);raise
    return str(target)


def sample_code(path, mode='chat'):
    operation="model.chat([{'role': 'user', 'content': 'Hello'}], 32)" if mode=='chat' else "model.predict(json.load(open('inputs.json')))"
    return ('import json, time\nfrom fxbex import '+('LLM' if mode=='chat' else 'Model')+'\n\n'
            f'PACKAGE = {str(path)!r}  # Change this filename\n'
            f'with {"LLM" if mode=="chat" else "Model"}(PACKAGE) as model:  # Load; release on exit\n'
            '    started = time.perf_counter()\n'
            f'    result = {operation}\n'
            '    print(result)\n'
            '    print(f"Inference: {time.perf_counter() - started:.3f}s")\n')


def build_ui(service_factory=None, default_package="", package_dir="fxbex", controls=None, mode='chat'):
    import gradio as gr
    from .service import PackageService
    service_factory=service_factory or PackageService
    directory=Path(package_dir).resolve();directory.mkdir(parents=True,exist_ok=True)
    packages=lambda:sorted(str(p) for p in directory.glob('*.fxbex'))
    state={'session':None};lock=threading.RLock()
    def unload():
        with lock:
            if state['session']:
                state['session'].unload();state['session']=None
            return 'Unloaded',[],[]
    def load(path,progress=gr.Progress()):
        with lock:
            unload();started=time.perf_counter();progress(0,desc='Checking package and loading NPU model')
            session=service_factory(path)
            try:session.load()
            except Exception:
                session.stop();raise
            state['session']=session;progress(1,desc='Ready')
            return f"Loaded: {path}\nLoad: {time.perf_counter()-started:.3f}s\nLog: {session.log_path}",[],[]
    def run(method,*args):
        with lock:
            if not state['session']:raise gr.Error('Load a package first')
            if method not in ('chat','predict'):raise ValueError('Unsupported operation')
            started=time.perf_counter()
            result=getattr(state['session'],method)(*args)
            return result,f"Request: {time.perf_counter()-started:.3f}s\n{json.dumps(result.get('metrics',{}),ensure_ascii=False)}"
    def chat(text,history):
        history=list(history or [])
        result,timing=run('chat',history+[{'role':'user','content':text}],32)
        history += [{'role':'user','content':text},{'role':'assistant','content':result['text']}]
        return '',history,history,timing
    gr.Markdown(f'# FXBEX model sample\nTrusted packages: `{directory}`. Load once, run repeatedly, then unload.')
    selected=gr.Dropdown(label='Saved packages',choices=packages(),value=None)
    refresh=gr.Button('Refresh packages')
    path=gr.Textbox(label='Package path on server',value=default_package)
    selected.change(lambda p:p,selected,path)
    refresh.click(lambda:gr.Dropdown(choices=packages()),outputs=selected)
    upload=gr.File(label='Save a trusted package to fxbex/',file_types=['.fxbex'],type='filepath')
    upload.change(lambda p:save_package(p,directory) if p else '',upload,path)
    with gr.Row():load_button=gr.Button('Load');unload_button=gr.Button('Unload')
    status=gr.Textbox(label='Status / load time / report')
    history=gr.State([])
    chatbot=gr.Chatbot(visible=controls is None and mode=='chat')
    if controls is not None:
        controls(run)
    elif mode=='chat':
        clear=gr.Button('Clear conversation');prompt=gr.Textbox(label='Message');send=gr.Button('Send')
        metrics=gr.Textbox(label='Execution time (wall clock, not NPU-only)')
        clear.click(lambda:([],[]),outputs=[history,chatbot],api_name='clear_conversation')
        send.click(chat,[prompt,history],[prompt,history,chatbot,metrics],api_name='chat_package')
    else:
        inputs=gr.JSON(label='Named tensor inputs');output=gr.JSON(label='Result');metrics=gr.Textbox(label='Execution time')
        gr.Button('Run').click(lambda x:run('predict',x),inputs,[output,metrics],api_name='predict_package')
    load_button.click(load,path,[status,history,chatbot],api_name='load_package')
    unload_button.click(unload,outputs=[status,history,chatbot],api_name='unload_package')
    gr.Markdown('## Python sample\nInstall FXBEX and the matching model adapter. Run on the SDK/NPU host. Change PACKAGE; tensor inputs follow the model contract. Unload the web model before running this example.')
    code=gr.Code(value=sample_code(default_package,mode),language='python',label='Load → run → release')
    path.change(lambda p:sample_code(p,mode),path,code)
    return unload


def launch(port=7861):
    import gradio as gr
    with gr.Blocks(title='FXBEX') as app:unload=build_ui()
    app.unload(unload)
    try:app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=port)
    finally:unload()
