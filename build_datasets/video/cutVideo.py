import getVideos
import json
from subprocess import call
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os

#result = getVideos.test("./videos.json")
#['001','002','003']
result = []
with open ("./videos.json","r") as loadJson:
    LOAD = json.load(loadJson)
    for key, value in LOAD.items():
        result.append(key)


#WordJson은 어절단위로 자른 영상 정답 라벨을 모은 제이슨파일
WordJson = []

for key in result:
	WordJson = [] #reset
	os.makedirs("./data/{}/video".format(key))
	os.makedirs("./data/{}/align".format(key))
	# data/001/video/001_1_001.avi
	# data/001/image/001_1_001/1.png
	VideoNameJson = {"VideoName" : key, "words":[]}
	WordJson.append(VideoNameJson)
	ListJson = []
	with open('/home/sej/STT-DataPreprocessing/STT/wavs/' + key + '_cut.json', encoding='cp949') as f:
		data = json.load(f)

	for count in range(len(data)):
		dictTime = data[count]['words']
		num = 0
		# data/001/video/001_1_001.avi
		# data/001/image/001_1_001/1.png
		
		for row in dictTime:
			# end_time, start_time, word를 WordTimeList안에 따로 담기  
			WordTimeList = []
			for stt in row.values():    
				WordTimeList.append(stt)
				num += 1
				#print(num/3)
				#print(WordTimeList)
			
			duration = WordTimeList[0] - WordTimeList[1]
			Num = str(int(num/3)).zfill(2)
			video_name = '{}_{}_{}'.format(key, count, Num)
			
			# 너무 짧고 긴 영상은 자르기
			if duration > 0.5 and duration < 1.3:
			
				#WordListJson[video_name] = WordTimeList[2]
				WordListJson = {"fileName": video_name, "word": WordTimeList[2], "duration": duration}
				ListJson.append(WordListJson)

				print(WordTimeList[2])
				#앞뒤로 0.3초 패딩
				ffmpeg_extract_subclip("./avi/{}.avi".format(key), 
										WordTimeList[1]-0.1, 
										WordTimeList[0]+2.0, 
										targetname="./data/{}/video/{}.avi".format(key, video_name))
				#align.txt 만들기
				txt = open("./data/{}/align/{}.txt".format(key, video_name), 'a')
				txt.write("word: {} \nduration: {} \nstarttime: {} \nendtime: {} ".format(WordTimeList[2], duration+0.5, WordTimeList[1]-0.3, WordTimeList[0]+0.2))
				txt.close	
		VideoNameJson["words"] = ListJson	

	with open('./data/{}/WordJson.json'.format(key), 'w') as f:
		json.dump(WordJson, f, indent=4, ensure_ascii=False)