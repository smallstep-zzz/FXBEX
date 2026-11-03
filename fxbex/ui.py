"""Optional package UI on a separate port; process owns one resident package."""
import threading


def build_ui(service_factory=None, default_package=""):
    import gradio as gr
    from .service import PackageService
    service_factory=service_factory or PackageService
    state={'session':None};lock=threading.RLock()
    def unload():
        with lock:
            if state['session']:
                state['session'].unload();state['session']=None
            return 'Unloaded',[],[]
    def load(path,progress=gr.Progress()):
        with lock:
            unload();progress(0,desc='Checking package and loading NPU model')
            session=service_factory(path)
            try:session.load()
            except Exception:
                session.stop();raise
            state['session']=session
            progress(1,desc='Ready')
            return f"Loaded: {path}\nLog: {session.log_path}",[],[]
    def chat(text,history):
        with lock:
            if not state['session']:raise gr.Error('Load a package first')
            history=list(history or [])
            result=state['session'].chat(history+[{'role':'user','content':text}],32)
            history += [{'role':'user','content':text},{'role':'assistant','content':result['text']}]
            return '',history,history,str(result.get('metrics',{}))
    gr.Markdown('# FXB_EX\nLoad trusted local `.fxbex` packages. Unload other NPU models first.')
    path=gr.Textbox(label='Package path on server',value=default_package)
    upload=gr.File(label='Or upload a trusted package',file_types=['.fxbex'],type='filepath')
    upload.change(lambda p:p,upload,path)
    with gr.Row():load_button=gr.Button('Load');unload_button=gr.Button('Unload')
    status=gr.Textbox(label='Status / report')
    history=gr.State([])
    chatbot=gr.Chatbot()
    clear=gr.Button("Clear conversation")
    prompt=gr.Textbox(label='Message')
    send=gr.Button('Send')
    metrics=gr.Textbox(label='Timing')
    load_button.click(load,path,[status,history,chatbot],api_name="load_package")
    unload_button.click(unload,outputs=[status,history,chatbot],api_name="unload_package")
    clear.click(lambda:([],[]),outputs=[history,chatbot],api_name='clear_conversation')
    send.click(chat,[prompt,history],[prompt,history,chatbot,metrics],api_name="chat_package")
    return unload


def launch(port=7861):
    import gradio as gr
    with gr.Blocks(title="FXBEX") as app:
        unload=build_ui()
    try:app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=port)
    finally:unload()
