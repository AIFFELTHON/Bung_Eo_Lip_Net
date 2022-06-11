import os
import json
import shutil
import sys


arg = sys.argv[1]

try:
    os.makedirs("./MostWord{}".format(arg))
except:
    pass

result=[]
with open ("./videos.json","r") as loadJson:
	LOAD = json.load(loadJson)
for key, value in LOAD.items():
	result.append(key)

#가지고 오고 싶은 단어 리스트안에 담기
mostWord = ['그리고', '뉴스에서', '하지만', '코로나', '코로나-19', '다른', '이렇게', '특히', '확진자', '한국', '한국어', '어떤', '이번', '오늘', '미국', '함께', '안녕하십니까', '여러분']

for file_name in mostWord:
    try:
        os.makedirs("./MostWord{}/{}/video".format(arg, file_name)) #폴더 이름 생각
        os.makedirs("./MostWord{}/{}/videoA".format(arg, file_name))
        os.makedirs("./MostWord{}/{}/align".format(arg, file_name))
    except:
        pass    
for key in result:
    txt_names = os.listdir('./data/{}/align/'.format(key))
    path ='./data/{}/align/'.format(key)
    #path = './data/A/face_video_align/
    #txt_names = os.listdir(path)
    

    for word in txt_names:
        txt_path = path + word
        wordList = open(txt_path, 'r')
        line = wordList.readline()
        #print(line[6:-2])
        try:
            if line[6:-2] in mostWord:
                txt = "파일명: " + word + line
                wordListtxt = open("word_list.txt", 'a')
                wordListtxt.write(txt)
                print(txt)
                shutil.copy(txt_path, "./MostWord{}/{}/align/{}.txt".format(arg, line[6:-2], word[:-4]))
                face_avi_path = txt_path.replace('align', 'face_video').replace('txt', 'avi')
                avi_path = txt_path.replace('align', 'video').replace('txt', 'avi') #face_vide - video ,video - videoA
                shutil.copy(face_avi_path, "./MostWord{}/{}/video/{}.avi".format(arg, line[6:-2], word[:-4]))
                shutil.copy(avi_path, "./MostWord{}/{}/videoA/{}.avi".format(arg, line[6:-2], word[:-4]))
            wordListtxt.close()
        except:
            print('error')
