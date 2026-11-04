"""Original model identity; never infer provenance from a display name."""
import re
from urllib.parse import urlsplit

def source_metadata(source_url=None, revision=None, existing=None):
    data=dict(existing or {})
    url=source_url if source_url is not None else data.get('source_url')
    if url:
        if re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',url):url='https://huggingface.co/'+url
        p=urlsplit(url)
        if p.scheme!='https' or not p.hostname or p.username or p.password or p.query or p.fragment:
            raise ValueError('Original model source must be an HTTPS URL without credentials/query/fragment')
        if p.hostname=='huggingface.co' and not re.fullmatch(r'/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?',p.path):raise ValueError('HF source must point to a model repository')
        data['source_url']=url.rstrip('/');data['status']='recorded'
    else:data.update(source_url=None,status='unknown')
    if revision is not None:data['revision']=revision
    if data.get('revision') is not None and (not isinstance(data['revision'],str) or not 1<=len(data['revision'])<=128):raise ValueError('Invalid source revision')
    return data
