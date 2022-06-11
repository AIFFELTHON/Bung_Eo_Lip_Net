
# https://github.com/khazit/Lip2Word/blob/master/lipReader.py

import os
import cv2
import numpy as np

# 영상 30프레임으로 맞추기 // 패딩

def videoToArray(path) :
    vidObj = cv2.VideoCapture(path)

    width = int(vidObj.get(3))
    height = int(vidObj.get(4))
    fps = int(vidObj.get(5))
    n_frames = int(vidObj.get(7))
    print("Video info : {}x{}, {} frames".format(height,width,n_frames))

    video = np.zeros((height, width, n_frames))
    video = video.astype(np.uint8)

    i = 0
    while True :
        success, frame = vidObj.read()
        if not success :
            break;
        else :
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video[:, :, i] = frame
            i += 1
    return video, n_frames, fps

def frameAdjust(video):
    #30 프레임 맞추기
    target = 30
    n_frames = video.shape[2]
    if target == n_frames : #30이면 그대로 리턴
        print("Perfect number of frames !")
        return video
    else :
        if n_frames > target : #30보다 크면 일정한 간격으로 맞춰 자르기
            print("Adjusting number of frames")
            idx = np.linspace(0, n_frames-1, 30) #1차원 배열만들기 : 0~ 총프레임 수까지 29개 일정한 간격으로 만들기
            idx = np.around(idx, 0).astype(np.int32) #반올림
            print("Indexes of the selected frames : \n{}".format(idx))
            return video[:, :, idx]
        else : #30보다 작으면 맨뒤의 프레임 동일하게 복제
            output_video = np.zeros((video.shape[0], video.shape[1], 30)).astype(np.uint8)
            output_video[:, :, :n_frames] = video
            for i in range(target-n_frames+1) :
                output_video[:, :, i+n_frames-1] = output_video[:, :, n_frames-1] #복제
            return output_video

def reshapeAndConvert(video) :
    size = video.shape[0]
    n_frames = video.shape[2]
    # (1, 150, 150, 30)
    video = np.reshape(video, (1, size, size, n_frames)).astype(np.float32)
    return video / 255.0

# Debugging function
# 일련의 프레임을 동영상으로 저장
def _write_video(video, path, name, fps) :
    #저장할 파일이름, fourcc, 초당 프레임수, 프레임크기
    writer = cv2.VideoWriter(
        path + name ,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (256,256),
        isColor=None)
    video = video * 255
    for i in range(30) :
        writer.write(
            cv2.resize(
                cv2.cvtColor(
                    video[0, :, :, i].astype('uint8'),
                    cv2.COLOR_GRAY2BGR
                ),
                dsize=(256, 256),
                interpolation=cv2.INTER_LINEAR
            )
        )
    writer.release()


#MostWord=['and','atnews','but','covid','differ','eruke','especially','huak','korea','korean'
#    ,'othun','this','today','usa','with']
MostWord = ['and', 'atnews']
for key in MostWord:  
    PATH = './MostWord/{}/'.format(key)
    dirPATH = PATH +'video/'
    os.makedirs(PATH +"videof30")
    file_names = os.listdir(dirPATH)
    for name in file_names:
        video_B, n_frames, fps = videoToArray('./MostWord/{}/video/{}'.format(key, name))
        video = frameAdjust(video_B)
        revideo=reshapeAndConvert(video)
        
        _write_video(revideo, PATH +"videof30/" ,name, fps)