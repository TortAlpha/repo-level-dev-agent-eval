"""Cache public benchmark images with isolated anonymous Docker configuration."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

from preflight import PRIMARY, ROOT, PLAN

REGISTRY = 'jefzda/sweap-images'

def main(tasks):
    rows = {r['task_id']: r for r in csv.DictReader((ROOT/'repositories/collection.csv').open())}
    endpoint = subprocess.check_output(['docker','context','inspect','--format','{{.Endpoints.docker.Host}}'], text=True).strip()
    docker_config = PLAN/'anonymous_docker'
    docker_config.mkdir(exist_ok=True)
    (docker_config/'config.json').write_text('{}\n')
    environment = {**os.environ, 'DOCKER_CONFIG': str(docker_config), 'DOCKER_HOST':endpoint}
    with urllib.request.urlopen(f'https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REGISTRY}:pull', timeout=20) as response:
        token = json.load(response)['token']
    records=[]
    for task_id in tasks:
        image = rows[task_id]['docker_image']
        tag = image.split(':',1)[1]
        request=urllib.request.Request(f'https://registry-1.docker.io/v2/{REGISTRY}/manifests/{tag}', headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.docker.distribution.manifest.v2+json'})
        with urllib.request.urlopen(request,timeout=20) as response:
            manifest=json.load(response)
            digest=response.headers.get('Docker-Content-Digest')
        records.append({'task_id':task_id,'image':image,'digest':digest,'manifest':manifest})
    layers={layer['digest']:layer['size'] for record in records for layer in record['manifest']['layers']}
    compressed=sum(layers.values())
    free=shutil.disk_usage(ROOT).free
    manifest_path=PLAN/('image_manifests_'+tasks[0].split('_')[1]+'.json')
    manifest_path.write_text(json.dumps({'unique_compressed_bytes':compressed,'free_bytes_before':free,'images':records},indent=2)+'\n')
    print(json.dumps({'unique_compressed_GiB':compressed/1024**3,'free_GiB':free/1024**3}),flush=True)
    # Earlier cached blocks used about 2.9 bytes on disk per compressed byte.
    # Keep an additional 4 GiB free, and recheck before each pair of pulls.
    if compressed*3+4*1024**3 > free:
        raise RuntimeError('Conservative disk guard: image preparation needs more free space.')
    logs=PLAN/'image_logs'
    logs.mkdir(exist_ok=True)
    def pull(record):
        if shutil.disk_usage(ROOT).free < 4*1024**3:
            raise RuntimeError('Disk guard: less than 4 GiB free.')
        with (logs/(record['task_id']+'.log')).open('w') as log:
            result=subprocess.run(['docker','pull',record['image']],env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        if result.returncode:
            raise RuntimeError(f"Image pull failed: {record['task_id']}; inspect its log.")
        inspected=json.loads(subprocess.check_output(['docker','image','inspect',record['image']],env=environment,text=True))[0]
        record['local']={k:inspected[k] for k in ['Id','RepoDigests','Architecture','Os','Size']}
        if REGISTRY+'@'+record['digest'] not in inspected['RepoDigests']:
            raise RuntimeError(f"Image digest changed: {record['task_id']}")
        print(json.dumps({'cached':record['task_id'],'size_GiB':inspected['Size']/1024**3}),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(pull, records))
    manifest_path.write_text(json.dumps({'unique_compressed_bytes':compressed,'free_bytes_before':free,'images':records},indent=2)+'\n')
    print('All selected images cached.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tasks',nargs='+',default=PRIMARY)
    main(parser.parse_args().tasks)
