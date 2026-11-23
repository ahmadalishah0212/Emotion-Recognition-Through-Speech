"""Resume one optional actor archive using four verified HTTP byte ranges."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5
from pathlib import Path
import argparse
import urllib.request
import shutil

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('actor',choices=['15']);args=parser.parse_args()
    size=525205576;expected='4b3b9bc8473e86f884630f2506684887'
    target=ROOT/f'data/cache/Video_Speech_Actor_{args.actor}.zip'
    offset=target.stat().st_size if target.exists() else 0
    if offset==size and md5(target.read_bytes()).hexdigest()==expected:return
    if offset>=size:raise ValueError('Unexpected archive size')
    url=f'https://zenodo.org/api/records/1188976/files/{target.name}/content'
    chunk=(size-offset+3)//4
    segments=[(i,offset+i*chunk,min(size-1,offset+(i+1)*chunk-1)) for i in range(4) if offset+i*chunk<size]
    def fetch(segment):
        i,start,end=segment;part=target.with_suffix(f'.part{i}')
        request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request,timeout=90) as response:
            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise ValueError('Invalid range response')
            with part.open('wb') as output:shutil.copyfileobj(response,output)
        if part.stat().st_size!=end-start+1:raise ValueError('Incomplete range')
        print('Downloaded range',i,flush=True)
        return part
    with ThreadPoolExecutor(max_workers=4) as pool:parts=list(pool.map(fetch,segments))
    with target.open('ab') as stream:
        for part in parts:
            with part.open('rb') as source:shutil.copyfileobj(source,stream)
    if target.stat().st_size!=size or md5(target.read_bytes()).hexdigest()!=expected:raise ValueError('Official checksum mismatch')
    for part in parts:part.unlink()
    print('Verified',target,flush=True)

if __name__=='__main__':main()
