"""Drive the real CLI and OpenAI SDK; never print prompts or generated answers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from openai import OpenAI, BadRequestError

package, baseline, report = map(Path, sys.argv[1:4])
report.mkdir(parents=True, exist_ok=False)
prior = json.loads(baseline.read_text())['baseline'][0]
expected = prior['first'], prior['second']
first = [{'role': 'user', 'content': 'Remember this context: '+('blue sky and green trees. '*10)+' What is 2+2? Reply only with the number.'}]
args = [sys.executable, '-m', 'fxbex', 'serve', str(package), '--port', '18002', '--report-dir', str(report/'runtime')]
start = time.perf_counter()
with (report/'server-stderr.log').open('w') as err:
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=err, text=True, env=os.environ.copy())
    try:
        for line in proc.stdout:
            if 'FXB_EX serving ' in line:
                break
        else:
            raise RuntimeError(f'server exited before readiness: {proc.wait()}')
        def drain():
            with (report/'server-stdout.log').open('w') as log:
                for line in proc.stdout:
                    log.write(line)
        drain_thread=threading.Thread(target=drain,daemon=True)
        drain_thread.start()
        load = time.perf_counter()-start
        client = OpenAI(api_key='local', base_url='http://127.0.0.1:18002/v1', timeout=300, max_retries=0)
        name = 'RedHatAI/Qwen3.5-9B-FP8-dynamic'
        assert name in [x.id for x in client.models.list().data]
        def chat(messages):
            begin=time.perf_counter()
            response=client.chat.completions.create(model=name,messages=messages,max_tokens=4)
            return response, time.perf_counter()-begin
        one,t1=chat(first)
        history=first+[{'role':'assistant','content':one.choices[0].message.content},{'role':'user','content':'Multiply that answer by 3. Reply only with the number.'}]
        two,t2=chat(history)
        match = (one.choices[0].message.content,two.choices[0].message.content)==expected
        assert match
        try:
            client.chat.completions.create(model=name,messages=first,temperature=0.8,max_tokens=4)
            raise AssertionError('unsupported option was accepted')
        except BadRequestError:
            pass
        metrics = two.model_extra.get('fxbex',{}).get('metrics',{})
        result={'outputs_match_previous':match,'server_load_s':load,'request_seconds':[t1,t2],
                'second_turn_reused_tokens':metrics.get('reused_input_tokens'),
                'unsupported_sampling_rejected':True,'openai_client_passed':True}
        (report/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        client.close()
    finally:
        proc.terminate()
        try:proc.wait(timeout=15)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
