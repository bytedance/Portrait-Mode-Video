# Portrait-Mode Video Recognition

We are releasing our dataset regarding Portrait-Mode Video Recognition research. The videos are sourced from [Douyin platform](https://www.douyin.com). We distribute video content through the provision of links. Users are responsible for downloading the videos independently. 

## Dataset
We provide links to ur videos in `video_links.csv`.

## Usage
For each of the links, you can download the video by using `requests` package. We also provide demo code in `download_videos.py`, where you will need to replace `target_dir` with your actual storage directory. 
```ruby
import requests
import os

def download(url, out):
    r = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }, allow_redirects=True, timeout=30, stream=True)
    assert r.headers["Content-Type"] == 'video/mp4', f"Content-Type is {r.headers['Content-Type']}, {r.content}"
    try:
        if not os.path.exists(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
    except Exception as e:
        print(e)
    open(out, "wb").write(r.content)

download(video_link, video_name)
```

## License
Our data is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/). The data is released for non-commercial research purposes only. By engaging in this downloading process, users are considered to have agreed to comply with our distribution license terms and conditions.
