"""Run official LLM wrapper on RNGD, reporting timings and booleans only."""
import json
from pathlib import Path
import sys
import time
from furiosa_llm import LLM, SamplingParams
from fxbex import hug

package, baseline, report = map(Path, sys.argv[1:4])
prior = json.loads(baseline.read_text())['baseline'][0]
expected = prior['first'], prior['second']
first = [{'role': 'user', 'content': 'Remember this context: '+('blue sky and green trees. '*10)+' What is 2+2? Reply only with the number.'}]
start=time.perf_counter()
model=hug(LLM)(model=str(package))
load=time.perf_counter()-start
try:
    params=SamplingParams(temperature=0,max_tokens=4)
    begin=time.perf_counter(); generated=model.generate('Say hello.',params); generation=time.perf_counter()-begin
    assert generated.outputs and generated.outputs[0].token_ids
    one=model.chat(first,params)[0]
    history=first+[{'role':'assistant','content':one.outputs[0].text},{'role':'user','content':'Multiply that answer by 3. Reply only with the number.'}]
    begin=time.perf_counter();two=model.chat(history,params)[0];second=time.perf_counter()-begin
    match=(one.outputs[0].text,two.outputs[0].text)==expected
    assert match
    result={'generate_passed':True,'chat_history_matches_previous':match,'load_s':load,
            'generate_s':generation,'second_chat_s':second,'output_tokens':generated.num_output_tokens}
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
finally:
    model.shutdown()
