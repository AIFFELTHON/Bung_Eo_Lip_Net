import json
from pytube import YouTube
from subprocess import call
from pydub import AudioSegment
import os

result = []

def test(jsonDir):
	# videos.json  읽어오기
    with open ("./videos.json","r") as loadJson:
        LOAD = json.load(loadJson)
    for key, value in LOAD.items():
        print(key)
        print(value)
	# youtube 영상 하나씩 받기 
        yt = YouTube('"{}"'.format(value))
	# youtube mp4로 (영상만)
    # 영상+음원은 adaptive=True
        stream = yt.streams.filter(adaptive=True, file_extension='mp4', only_video=True).order_by('resolution').desc().first()
        stream.download('./', '{}.mp4'.format(key))

	# mp4를 avi로
        call('ffmpeg -i ./{}.mp4 ./avi/{}.avi'.format(key, key), shell=True)

	# mp4 제거
        os.remove("{}.mp4".format(key))
        result.append(key)

    return result 
