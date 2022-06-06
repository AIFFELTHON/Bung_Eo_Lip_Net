import os
import json

# with open ("./videos.json","r") as loadJson:
#     LOAD = json.load(loadJson)
#     for key, value in LOAD.items():
#         result.append(key) 

def makeImageDir(key):
    image_path= []
    file_names = os.listdir('./data/{}/video'.format(key))
    for label in file_names:
        t_label=label[:-4]
        #os.makedirs("./data/{}/image/{}".format(key, t_label)) #label
        mp4_path = './data/{}/video/'.format(key) + label
        image_path.append(mp4_path)
    return image_path