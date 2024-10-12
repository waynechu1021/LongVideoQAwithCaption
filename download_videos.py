import argparse, os
# parser = argparse.ArgumentParser()
# parser.add_argument('--source_path', required=True, type=str)
# parser.add_argument('--video_path', required=True, type=str)
# args = parser.parse_args()

# source_path = args.source_path
# video_path = args.video_path
source_path = './playground/Moment-10M_1.json'
video_path = '/hdd2/zwy/Momentor_video'

import json, numpy as np, copy
from tqdm import tqdm

print(f'Loading data from {source_path}')

with open(source_path, 'r') as f:
    packed_data = json.load(f)
    
print('Start downloading.')
    
video_names = list(packed_data.keys())
youtube_video_format = 'https://www.youtube.com/watch?v={}'
video_path_format = os.path.join(video_path, '{}.mp4')

for video_name in video_names:
    try:
        url = youtube_video_format.format(video_name)
        file_path = video_path_format.format(video_name)
        if os.path.exists(file_path):
            continue
        os.system("yt-dlp --username oauth2 --password '' -o " + file_path + ' -f 134 ' + url)
        print(f'Downloading of Video {video_name} has finished.')
    except:
        print(f'Downloading of Video {video_name} has failed.')
        
print('Finished.')
