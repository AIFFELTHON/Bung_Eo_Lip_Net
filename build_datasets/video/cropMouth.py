import os
import json
import dlib
import skvideo.io
import cv2
import makeDir

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


with open ("./videos.json","r") as loadJson:
    LOAD = json.load(loadJson)
    for key, value in LOAD.items():
        #return key

        # 폴더 만들기
        # 이미지 경로 불러오기
        image_path = makeDir. makeImageDir(key)
        print(image_path)

        ## 비디오 읽어오기
        for mp4_path in image_path:
            print("this Count session = " + mp4_path)
            v_cap = cv2.VideoCapture(mp4_path)
            if v_cap.isOpened():
                while True:
                    success, image = v_cap.read()  # BGR
                    if success :
                    # if success and int(v_cap.get(1)) % 5 == 0:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #흑백
                        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        resized = cv2.resize(image, dsize=(1600, 1200), interpolation=cv2.INTER_LINEAR)  # (1920, 1080)
                        rects = detector(resized, 1)
                        for i, rect in enumerate(rects):
                            shape = predictor(resized, rect)
                            left_x, left_y, right_x, right_y = 0, 0, 0, 0

                            for j in range(68):  # Face Detection
                            #for j in range(48, 68):  # Mouth Detection
                                x, y = shape.part(j).x, shape.part(j).y
                            left_x = shape.part(4).x
                            left_y = shape.part(30).y
                            right_x = shape.part(13).x
                            right_y = shape.part(8).y
                            resized = resized[left_y:right_y, left_x:right_x]  # [높이(행), 너비(열)]

                            PATH = mp4_path.replace('video', 'image')
                            cv2.imwrite(os.path.join(PATH[:-4], '%d.png' % v_cap.get(1)), resized)
                            #./data/001/001_0/image/001_01안녕하세요
                            #./data/001/001_0/image/001_0_01
                            #./data/001/001_0/video/001_0_01.avi
                            
                            print("Frame Captured: %d" % v_cap.get(1))
                            #if cv2.waitKey(1) & 0xFF == ord('q'):
                            #break
                        # cv2.destroyAllWindows()
                    else :
                        break
            