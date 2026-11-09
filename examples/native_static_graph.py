"""Build a real tiny FXB, then load it in another process with external weights."""
import argparse,json,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','run']);parser.add_argument('package',type=Path)
    args=parser.parse_args();weights=args.package.with_suffix('.weights.pt')
    import torch
    from fxbx import build_model,GraphFXB
    torch.manual_seed(71);x=torch.randn(4,8,dtype=torch.bfloat16)
    started=time.monotonic()
    if args.action=='build':
        if weights.exists():raise FileExistsError(weights)
        model=torch.nn.Linear(8,6).bfloat16().eval().requires_grad_(False)
        result=build_model(model,args.package,example_inputs=(x,))
        with weights.open('xb') as stream:torch.save(model.state_dict(),stream)
        print(json.dumps({'status':result['status'],'fxb':str(args.package.resolve()),'external_weights':str(weights.resolve()),'seconds':time.monotonic()-started}))
    else:
        state=torch.load(weights,weights_only=True)
        reference=torch.nn.functional.linear(x,state['weight'],state['bias'])
        with GraphFXB(args.package,weights=state) as model:y,=model.predict((x,))
        torch.testing.assert_close(y,reference,atol=.02,rtol=.02)
        print(json.dumps({'shape':list(y.shape),'max_abs_vs_cpu_oracle':float((y-reference).abs().max()),'seconds':time.monotonic()-started,'output':y.tolist()}))
if __name__=='__main__':main()
