"""Official API plus FXBEX: python examples/hug_llm.py ./model.fxbex"""
import asyncio,sys
from furiosa_llm import LLM,SamplingParams
from fxbex import hug

LLM=hug(LLM)
params=SamplingParams(temperature=0,max_tokens=16)
async def stream(model):
    async for delta in model.stream_generate('The numbers are one, two,',params):
        print(delta,end='',flush=True)
    print()

with LLM(sys.argv[1],max_model_len=512) as model:
    answers=model.chat([{'role':'user','content':'What is 2+2? Reply briefly.'}],params)
    print(answers[0].outputs[0].text)
    asyncio.run(stream(model))
