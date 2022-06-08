#coding=utf-8
import os
import cv2  # OpenCV 라이브러리
import numpy as np
from PIL import Image


# -- IO utils
# 텍스트 라인 불러오기
def read_txt_lines(filepath):
    # 파일이 있는지 확인, 없으면 AssertionError 메시지를 띄움
    assert os.path.isfile( filepath ), "Error when trying to read txt file, path does not exist: {}".format(filepath)  # 원하는 조건의 변수값을 보증하기 위해 사용
    
    # 파일 불러오기
    with open( filepath ) as myfile:
        content = myfile.read().splitlines()  # 문자열을 '\n' 기준으로 쪼갠 후 list 생성
    return content


# npz 저장
def save2npz(filename, data=None):
    # 데이터가 비어있는지 확인, 없으면 AssertionError 메시지를 띄움               
    assert data is not None, "data is {}".format(data)          
    
    # 파일 없을 경우                 
    if not os.path.exists(os.path.dirname(filename)):                            
        os.makedirs(os.path.dirname(filename))  # 디렉토리 생성
    np.savez_compressed(filename, data=data)  # 압축되지 않은 .npz 파일 형식 으로 여러 배열 저장


# 비디오 불러오기
def read_video(filename):
    cap = cv2.VideoCapture(filename)  # 영상 객체(파일) 가져오기

    while(cap.isOpened()):  # 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부  
        # ret: 정상적으로 읽어왔는가?
        # frame: 한 장의 이미지(frame) 가져오기
        ret, frame = cap.read() # BGR                                            
        if ret:  # 프레임 정보를 정상적으로 읽지 못하면                                                                  
            yield frame  # 프레임을 함수 바깥으로 전달하면서 코드 실행을 함수 바깥에 양보                                                        
        else:  # 프레임 정보를 정상적으로 읽지 못하면                                                                    
            break  # while 빠져나가기                                                                 
    cap.release()  # 영상 파일(카메라) 사용 종료
    


# Video 정보 가져오기
def get_video_info(infilename, is_print=False):
    cap = cv2.VideoCapture(infilename)
    if not cap.isOpened():
        print("could not open : ", infilename)
        cap.release()
        exit(0)
    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if is_print:
        print('length : ', length)
        print('width : ', width)
        print('height : ', height)
        print('fps : ', fps)
        
    video_info = {
        'length': length,
        'width': width,
        'height': height,
        'fps': fps,
    }
    
    return video_info

# Video -> Numpy
# 참고 깃허브 코드: https://github.com/khazit/Lip2Word/blob/master/lipReader.py#L22
def videoToArray(video_pathname, is_gray=True) :
    
    cap = cv2.VideoCapture(video_pathname)  # 영상 객체(파일) 가져오기
    
    # 영상 파일(카메라)이 정상적으로 열리지 않은 경우
    if not cap.isOpened():
        print("could not open : ", video_pathname)
        cap.release()  # 영상 파일(카메라) 사용 종료
        exit(0)  # 빠져나가기
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 영상 프레임 개수
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상 너비
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상 높이
    fps = cap.get(cv2.CAP_PROP_FPS)  # 영상 FPS(Frames Per Second)
    
    if is_gray:
        video = np.zeros((n_frames, height, width))  # gray
    else:
        n_channels=3
        video = np.zeros((n_frames, height, width, n_channels))  # color
        
    video = video.astype(np.uint8)
    
    i = 0
    while True :
        success, frame = cap.read()
        if not success :
            break
        else :
            # gray scale 적용
            if is_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            video[i] = frame
            i += 1
            
    cap.release()  # 영상 파일(카메라) 사용 종료
    
    return video  # 영상 정보 앞에 영상 프레임 개수를 추가한 numpy 반환


# Frame Sampling (프레임 개수 맞추기)
# 참고 깃허브 코드: https://github.com/khazit/Lip2Word/blob/master/lipReader.py#L62
def frameAdjust(video, target_frames=29):
    n_frames = video.shape[0]  # 영상 프레임 개수
    
    if target_frames == n_frames :
        return video  # 영상 그대로 반환
    else :
        # 영상 프레임 개수 > 원하는 프레임 개수
        if n_frames > target_frames :
            idx = np.linspace(0, n_frames-1, target_frames)  # 숫자 시퀀스 생성 # 구간 시작점, 구간 끝점, 구간 내 숫자 개수
            idx = np.around(idx, 0).astype(np.int32)  # 반올림하고 dtype 을 정수로 변경
            return video[idx]  # 원하는 프레임 개수로 sampling 한 영상
        # 영상 프레임 개수 < 원하는 프레임 개수
        else :
            output_video = np.zeros((target_frames, *video.shape[1:])).astype(np.uint8)  # 원하는 프레임 개수에 맞춰서 0으로 초기화한 numpy 생성
            output_video[:n_frames] = video  # 영상 프레임 개수까지 그대로 영상 정보 저장
            
            # 원하는 프레임 개수만큼 마지막 프레임 복제
            for i in range(target_frames-n_frames+1) :
                output_video[i+n_frames-1] = output_video[n_frames-1]
                
            return output_video  # 원하는 프레임 개수로 sampling 한 영상
