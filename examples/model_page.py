"""Run an existing model package in Gradio. Set PACKAGE before starting."""
from pathlib import Path
import gradio as gr
from fxbex.ui import build_ui

PACKAGE = Path('fxbex/model.fxbx')
MODE = 'chat'  # 'predict' for named tensor JSON; custom controls below for images etc.

# Agent extension: decode/encode the model's real inputs and outputs here.
# def model_controls(run):
#     inputs = gr.JSON(label='Model inputs')
#     output = gr.JSON(label='Model result')
#     timing = gr.Textbox(label='Execution time')
#     gr.Button('Run').click(lambda x: run('predict', x), inputs, [output, timing])

with gr.Blocks(title='fxbX model sample') as app:
    unload = build_ui(default_package=str(PACKAGE), package_dir=PACKAGE.parent,
                      mode=MODE)  # controls=model_controls for a custom model UI
app.unload(unload)
try:
    app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', server_port=7862)
finally:
    unload()
