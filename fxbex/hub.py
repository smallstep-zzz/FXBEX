"""Resolve catalog model IDs to immutable, checksum-verified HF packages."""
import hashlib,json,os,sys,shutil,tempfile
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlsplit

CATALOG='https://raw.githubusercontent.com/smallstep-zzz/furyawesome/main/catalog.json'

def resolve_package(value):
    path=Path(value).expanduser()
    if path.is_file():return path.resolve()
    if path.is_dir():
        installed=path/'model.fxbex'
        if installed.is_file():return installed.resolve()
    identifier=str(value)
    if identifier.startswith('./'):identifier=identifier[2:]
    if identifier.endswith('.fxbex') or identifier.startswith(('.', '/', '~')):raise FileNotFoundError(path)
    with urlopen(CATALOG,timeout=30) as response:
        raw=response.read(4_000_001)
    if len(raw)>4_000_000:raise ValueError('Catalog too large')
    catalog=json.loads(raw)
    name,sep,version=identifier.partition('@')
    matches=[p for p in catalog['models'] if name in (p['model']['name'],p['model']['source_repo']) and (not sep or p['release']['version']==version)]
    if len(matches)!=1:raise ValueError('Model ID is missing or ambiguous; use exact source_repo@version from FuryAwesome')
    release=matches[0]['release'];u=urlsplit(release['artifact_url']);parts=u.path.strip('/').split('/')
    import re
    if u.scheme!='https' or u.netloc!='huggingface.co' or u.query or u.fragment or len(parts)<5 or parts[2]!='resolve' or not re.fullmatch('[0-9a-f]{40}',parts[3]) or any(x in ('.','..','') for x in parts):raise ValueError('Expected immutable HF artifact URL')
    expected=release['author_sha256']
    if not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('Missing SHA256')
    try:from huggingface_hub import hf_hub_download
    except ImportError:raise RuntimeError('Install download support: pip install huggingface_hub') from None
    from tqdm.auto import tqdm
    class DownloadProgress(tqdm):
        def __init__(self,*args,**kwargs):
            kwargs['disable']=False
            super().__init__(*args,**kwargs)
    print('Downloading model from Hugging Face (cached files are reused)...',file=sys.stderr,flush=True)
    path=Path(hf_hub_download('/'.join(parts[:2]),'/'.join(parts[4:]),revision=parts[3],tqdm_class=DownloadProgress))
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected:raise ValueError('Downloaded package checksum mismatch')
    destination=Path(value).expanduser()
    destination.mkdir(parents=True,exist_ok=True)
    target=destination/'model.fxbex'
    if target.exists():raise FileExistsError(target)
    fd,temp=tempfile.mkstemp(prefix='.download-',dir=destination);os.close(fd)
    try:
        from tqdm.auto import tqdm
        with path.open('rb') as source,open(temp,'wb') as output,tqdm(total=path.stat().st_size,unit='B',unit_scale=True,desc='Saving model',file=sys.stderr) as bar:
            for block in iter(lambda:source.read(1024*1024),b''):
                output.write(block);bar.update(len(block))
        os.replace(temp,target)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    print('Model saved: '+str(target.resolve()),file=sys.stderr,flush=True)
    return target.resolve()
