import os
import json
import shutil


#가지고 오고 싶은 단어 리스트안에 담기
mostWord = ['그리고', '뉴스에서', '하지만', '코로나', '코로나-19', '다른', '이렇게', '특히', '확진자', '한국', '한국어', '어떤', '이번', '오늘', '미국', '함께', '안녕하십니까', '여러분']

for file_name in mostWord:
    try:
        os.makedirs("./MostWord/{}/video".format(file_name)) #폴더 이름 생각
        os.makedirs("./MostWord/{}/videoA".format(file_name))
        os.makedirs("./MostWord/{}/align".format(file_name))
    except:
        pass
path ='./data/A/face_video_align/'
txt_names = os.listdir(path)

    #path = './data/A/face_video_align/
    #txt_names = os.listdir(path)

for word in txt_names:
    txt_path = path + word
    wordList = open(txt_path, 'r')
    line = wordList.readline()
    #print(line[6:-2])
# txt -> avi
    try:
        if line[6:-2] in mostWord:
            txt = "파일명: " + word + line
            wordListtxt = open("word_list.txt", 'a')
            wordListtxt.write(txt)
            print(txt)
            face_avi_path = txt_path.replace('face_video_align', 'face_video').replace('txt', 'avi')
            avi_path = txt_path.replace('face_video_align', 'video').replace('txt', 'avi') #face_vide - video ,video - videoA
            #align_path = txt_path.replace('face_video_align', 'face_video')
            shutil.copy(face_avi_path, "./MostWord/{}/video/{}.avi".format(line[6:-2], word[:-4]))
            shutil.copy(avi_path, "./MostWord/{}/videoA/{}.avi".format(line[6:-2], word[:-4]))
            shutil.copy(txt_path, "./MostWord/{}/align/{}".format(line[6:-2], word))
        wordListtxt.close()
    except:
        print('error')