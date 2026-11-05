"""Resolve catalog model IDs to immutable, checksum-verified HF packages."""
import hashlib,json,os,sys,shutil,tempfile
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlsplit

CATALOG='https://raw.githubusercontent.com/smallstep-zzz/furyawesome/main/catalog.json'

def _local_package(path, *, announce=True):
    if path.is_file():return path.resolve()
    if not path.is_dir():return None
    candidates=[p for p in path.rglob('*.fxbex') if p.is_file()]
    if not candidates:return None
    selected=max(candidates,key=lambda p:(p.stat().st_mtime_ns,str(p)))
    if announce:print('Selected model: '+str(selected.resolve()),file=sys.stderr,flush=True)
    return selected.resolve()


def catalog_matches(catalog,name,version=None):
    matches=[p for p in catalog['models'] if name in (p['model']['name'],p['model']['source_repo'],p['model']['source_repo'].split('/')[-1]) and (version is None or p['release']['version']==version)]
    sources={p['model']['source_repo'] for p in matches}
    if len(sources)>1:raise ValueError('Short model name is ambiguous; include the publisher/model ID')
    if len(matches)<=1:return matches
    order=catalog.get('release_order',[])
    def rank(p):
        key=p['model']['source_repo']+'@'+p['release']['version']+'#'+p['release']['author_sha256']
        return order.index(key) if key in order else -1
    if max(map(rank,matches))<0:raise ValueError('No catalog release order available; select MODEL@VERSION')
    return [max(matches,key=rank)]


def resolve_package(value):
    import re
    identifier=str(value)
    if identifier.startswith('./'):identifier=identifier[2:]
    name,sep,version=identifier.partition('@')
    if sep and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,59}',version):raise ValueError('Invalid release version')
    path=Path(value).expanduser()
    explicit_local=isinstance(value,os.PathLike) or str(value).startswith(('.', '/', '~')) or identifier.endswith('.fxbex') or path.is_file()
    inferred=None
    if '/' not in name and not path.exists():
        folders=[p for p in Path('.').glob('*/'+name) if p.is_dir()]
        if len(folders)==1:inferred=folders[0]
        elif len(folders)>1:raise ValueError('Short model name is ambiguous locally; include publisher/model ID')
    # An explicit version never reuses an unversioned legacy file.
    candidate=inferred if inferred else path
    local=_local_package(candidate.parent/candidate.name.split('@')[0]/version if sep else candidate,announce=False)
    if local and (explicit_local or sep):
        print('Selected model: '+str(local),file=sys.stderr,flush=True)
        return local
    if explicit_local and (isinstance(value,os.PathLike) or identifier.endswith('.fxbex') or identifier.startswith(('.', '/', '~'))):raise FileNotFoundError(path)
    # Inferred publisher/model folders are caches, not explicit local selections.
    # A short catalog ID must not silently select an older, recently touched file.
    try:
        with urlopen(CATALOG,timeout=30) as response:
            raw=response.read(4_000_001)
    except OSError:
        if local:
            print('Catalog unavailable; using local package. Latest release is unknown.',file=sys.stderr,flush=True)
            print('Selected model: '+str(local),file=sys.stderr,flush=True)
            return local
        raise
    if len(raw)>4_000_000:raise ValueError('Catalog too large')
    catalog=json.loads(raw)
    name,sep,version=identifier.partition('@')
    matches=catalog_matches(catalog,name,version if sep else None)
    if not matches and local:
        print('Selected model: '+str(local),file=sys.stderr,flush=True)
        return local
    if len(matches)!=1:raise ValueError('Model ID is missing or ambiguous; use exact source_repo@version from FuryAwesome')
    release=matches[0]['release'];u=urlsplit(release['artifact_url']);parts=u.path.strip('/').split('/')
    import re
    if u.scheme!='https' or u.netloc!='huggingface.co' or u.query or u.fragment or len(parts)<5 or parts[2]!='resolve' or not re.fullmatch('[0-9a-f]{40}',parts[3]) or any(x in ('.','..','') for x in parts):raise ValueError('Expected immutable HF artifact URL')
    expected=release['author_sha256']
    if not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('Missing SHA256')
    source_id=matches[0]['model']['source_repo']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',source_id) or any(x in ('.','..') for x in source_id.split('/')):raise ValueError('Invalid source model ID')
    selected_version=release['version']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,59}',selected_version):raise ValueError('Invalid release version')
    print(f'Selected release: {source_id}@{selected_version}',file=sys.stderr,flush=True)
    destination=Path(source_id)/selected_version
    target=destination/(source_id.split('/')[-1]+'.fxbex')
    if target.exists():
        local_hash=hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):local_hash.update(block)
        if local_hash.hexdigest()!=expected:raise ValueError('Installed release differs from catalog checksum; refusing to overwrite')
        print('Selected model: '+str(target.resolve()),file=sys.stderr,flush=True)
        return target.resolve()
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
    destination.mkdir(parents=True,exist_ok=True)
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
