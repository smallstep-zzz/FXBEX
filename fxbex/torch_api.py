"""Explicit inference-only nn.Module, not a replacement for arbitrary library pipelines."""
import torch
from .runtime import Session


def restore(tree,values):
    if isinstance(tree,str):return values[tree]
    if isinstance(tree,list):return [restore(t,values) for t in tree]
    if isinstance(tree,dict) and set(tree)=={'constant'}:return tree['constant']
    if isinstance(tree,dict) and set(tree)=={'library_output'}:
        spec=tree['library_output'];fields={k:restore(v,values) for k,v in spec['fields'].items()}
        if spec['library']=='transformers':
            from transformers import modeling_outputs
            allowed={'BaseModelOutput','BaseModelOutputWithPoolingAndCrossAttentions','BaseModelOutputWithPastAndCrossAttentions','BaseModelOutputWithPast','ImageClassifierOutputWithNoAttention','ImageClassifierOutput','CausalLMOutputWithPast'}
            if spec['class'] not in allowed:raise NotImplementedError('Unregistered Transformers output class')
            return getattr(modeling_outputs,spec['class'])(**fields)
        if spec['library']=='diffusers':
            if spec['class']=='DecoderOutput':
                from diffusers.models.autoencoders.vae import DecoderOutput
                return DecoderOutput(**fields)
            if spec['class']=='UNet2DOutput':
                from diffusers.models.unets.unet_2d import UNet2DOutput
                return UNet2DOutput(**fields)
        raise NotImplementedError('Unregistered library output contract')
    if isinstance(tree,dict) and set(tree)=={'tuple'}:return tuple(restore(t,values) for t in tree['tuple'])
    if isinstance(tree,dict):return {k:restore(t,values) for k,t in tree.items()}
    raise ValueError('Invalid declared output structure')

class CompiledModule(torch.nn.Module):
    def __init__(self,package,*,execution_policy='npu_only',inference_only=True,output_device='cpu'):
        super().__init__()
        if execution_policy!='npu_only' or inference_only is not True:raise ValueError('Only NPU-only inference is supported')
        if output_device not in ('cpu','furiosa'):raise ValueError('Unsupported output device')
        self.output_device=output_device;self.session=Session(package).__enter__();self.closed=False
        try:
            if not hasattr(self.session.adapter,'predict_tensors'):raise NotImplementedError('Adapter has no tensor contract')
            self.config=self.session.adapter.config;self.names=list(self.config['inputs']);self.training=False
        except BaseException:self.session.__exit__(None,None,None);raise
    def forward(self,*args,**kwargs):
        if self.closed:raise RuntimeError('Compiled module is closed')
        if len(args)>len(self.names):raise TypeError('Too many positional inputs')
        values=dict(zip(self.names,args))
        if set(values)&set(kwargs):raise TypeError('Input provided twice')
        values.update(kwargs)
        if set(values)!=set(self.names):raise TypeError('Unexpected or missing compiled input; cache/sampling options require an explicit adapter')
        result=self.session.predict_tensors(values,output_device=self.output_device)['outputs']
        return restore(self.config['output_tree'],result) if 'output_tree' in self.config else result
    def train(self,mode=True):
        if mode:raise NotImplementedError('Compiled module is inference-only')
        self.training=False;return self
    def to(self,*args,**kwargs):
        raise NotImplementedError('Immutable compiled weights: select output_device when loading; dtype/device migration is unsupported')
    def _apply(self,fn,*args,**kwargs):raise NotImplementedError('Compiled dtype/device is immutable')
    def state_dict(self,*args,**kwargs):raise NotImplementedError('Weights are immutable package artifacts')
    def load_state_dict(self,*args,**kwargs):raise NotImplementedError('Rebuild and revalidate a new package to change weights')
    def close(self):
        if not self.closed:self.session.__exit__(None,None,None);self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
