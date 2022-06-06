# 폴더 / face_video_align / 
# 폴더 / face_video
import os
import json

txt_names = os.listdir('./data/{}/face_video_align'.format(key))

path ='align 모아놓은 폴더 경로'

lines=[]
for word in txt_names:
    txt_path = path + word
    wordList = open(txt_path, 'r')
    line = wordList.readline()
    lines.append(line[6:])
print(lines)


wordListtxt = open("word_list.txt", 'w')

for i in lines:
  wordListtxt.write(i)
wordListtxt.close()


def create_dict_word_list(path) :
    count = 0
    my_dict = dict()
    with open(path+'word_list.txt', 'r') as f:
        for line in f:
            my_dict.update({count : line[:-1]})
            count += 1
    return my_dict

with open('dict.json','w') as f:
    json.dump(my_dict, f, indent=4, ensure_ascii=False)