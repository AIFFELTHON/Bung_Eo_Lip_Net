import json
from pytube import YouTube
from subprocess import call
from pydub import AudioSegment
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="key.json"

result = []

def test(jsonDir):
	# videos.json  읽어오기
	with open (jsonDir,"r", encoding="UTF-8") as loadJson:
		LOAD = json.load(loadJson)

	# videos.json  Key,Value로 하나씩 돌리기 위한 for문 
	for key, value in LOAD.items():
		print(key)
		print(value)
	# youtube 음원 하나씩 받기 
		yt = YouTube('"{}"'.format(value))
	# youtube 음원 mp4로 (형식만 mp4)
		yt.streams.filter(only_audio=True).first().download('./', '{}.mp4'.format(key))

	# mp4를 wav로    
		call('ffmpeg -i ./{}.mp4 ./wavs/{}_a.wav'.format(key, key), shell=True)

	# wav파일 이름 지정 
		sound = AudioSegment.from_wav("./wavs/{}_a.wav".format(key))
	# wav파일 채널 변경 
		sound = sound.set_channels(1)
	# 변경후 저장 ` 
		sound.export("./wavs/{}.wav".format(key), format="wav")

	# mp4,wav 제거
		os.remove("{}.mp4".format(key))
		os.remove("./wavs/{}_a.wav".format(key))
		result.append(key+".wav")
	return result
