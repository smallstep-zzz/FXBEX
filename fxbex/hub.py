"""Resolve catalog model IDs to immutable, checksum-verified HF packages."""
import hashlib,json,os,sys,shutil,tempfile
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlsplit

CATALOG='https://raw.githubusercontent.com/smallstep-zzz/furyawesome/main/catalog.json'

def _local_package(path):
    if path.is_file():return path.resolve()
    if not path.is_dir():return None
    for filename in (path.name+'.fxbex','model.fxbex'):
        candidate=path/filename
        if candidate.is_file():return candidate.resolve()
    candidates=list(path.glob('*.fxbex'))
    if len(candidates)==1:return candidates[0].resolve()
    if len(candidates)>1:raise ValueError('Multiple model files; select the exact .fxbex path')
    versions=[p for p in path.glob('*/*.fxbex') if p.is_file()]
    if len(versions)==1:return versions[0].resolve()
    if len(versions)>1:raise ValueError('Multiple installed versions; select MODEL@VERSION or an exact file')
    return None


def resolve_package(value):
    import re
    identifier=str(value)
    if identifier.startswith('./'):identifier=identifier[2:]
    name,sep,version=identifier.partition('@')
    if sep and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,59}',version):raise ValueError('Invalid release version')
    path=Path(value).expanduser()
    # An explicit version never reuses an unversioned legacy file.
    local=_local_package(Path(name).expanduser()/version) if sep else _local_package(path)
    if local:return local
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
    source_id=matches[0]['model']['source_repo']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',source_id) or any(x in ('.','..') for x in source_id.split('/')):raise ValueError('Invalid source model ID')
    selected_version=release['version']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,59}',selected_version):raise ValueError('Invalid release version')
    destination=Path(source_id)/selected_version
    destination.mkdir(parents=True,exist_ok=True)
    target=destination/(source_id.split('/')[-1]+'.fxbex')
    if target.exists():
        local_hash=hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):local_hash.update(block)
        if local_hash.hexdigest()!=expected:raise ValueError('Installed release differs from catalog checksum; refusing to overwrite')
        return target.resolve()
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
