"""
Copyright 2023 Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import requests
import csv
import os
import glob

import multiprocessing as mp


urls = []
failed = []
target_dir = './data'

all_vids = glob.glob(f'{target_dir}/*mp4')
all_vnames = [p.split('/')[-1][:-4] for p in all_vids]
with open('existing_videos.csv', 'w') as f:
    f.writelines('\n'.join(all_vnames))

with open('existing_videos.csv', 'r') as f:
    existing_vids = [l.strip() for l in f.readlines()]


def download(url):
    vname = url.split('=')[-1]+'.mp4'
    out = os.path.join(target_dir, vname)
    if vname[:-4] in existing_vids:
        return True, vname

    try:
        r = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }, allow_redirects=True, timeout=30, stream=True)
        assert r.headers["Content-Type"] == 'video/mp4', f"Content-Type is {r.headers['Content-Type']}, {r.content}"
        if not os.path.exists(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
        open(out, "wb").write(r.content)
        
        if os.path.getsize(out)/1024 <= 20:
            raise AssertionError
    except Exception as e:
        print(e)
        return False, vname, url

    return True, vname, url


urls = []
for url in csv.reader(open('video_links.csv', 'r')):
    urls.append(url[0])

failed = []
pool = mp.Pool(100)
res = pool.map(download, urls)
pool.close()
pool.join()

for r in res:
    if not r[0]:
        failed.append(r[2])

with open('fail_cases.csv', 'w') as f:
    f.writelines('\n'.join(failed))
