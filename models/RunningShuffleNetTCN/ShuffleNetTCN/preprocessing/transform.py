import cv2  # OpenCV 라이브러리
import numpy as np                                                               
from skimage import transform as tf  # 이미지 변환 모듈

# -- Landmark interpolation:
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]  # 랜드마크 시작
    stop_landmarks = landmarks[stop_idx]  # 랜드마크 끝
    delta = stop_landmarks - start_landmarks  # 랜드마크 값 차이
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta  # 랜드마크 업데이트(보간)
    return landmarks

# -- Face Transformation
# src: 입력 영상, dst: 출력/결과 영상
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix  # 변환 행렬 구하기
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # wrap the frame image  # 주어진 좌표 변환에 따라 프레임 이미지 왜곡
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')  # numpy 데이터 타입 uint8 으로 변경
    return warped, tform

def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)  # wrap the frame image  # 주어진 좌표 변환에 따라 프레임 이미지 왜곡
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')  # numpy 데이터 타입 uint8 으로 변경
    return warped

# -- Crop
def cut_patch(img, landmarks, height, width, threshold=5):

    center_x, center_y = np.mean(landmarks, axis=0)  # 각 그룹의 같은 원소끼리 평균

    # 좌표 처리
    if center_y - height < 0:                                                
        center_y = height                                                    
    if center_y - height < 0 - threshold:                                    
        raise Exception('too much bias in height')                           
    if center_x - width < 0:                                                 
        center_x = width                                                     
    if center_x - width < 0 - threshold:                                     
        raise Exception('too much bias in width')                            
                                                                             
    if center_y + height > img.shape[0]:                                     
        center_y = img.shape[0] - height                                     
    if center_y + height > img.shape[0] + threshold:                         
        raise Exception('too much bias in height')                           
    if center_x + width > img.shape[1]:                                      
        center_x = img.shape[1] - width                                      
    if center_x + width > img.shape[1] + threshold:                          
        raise Exception('too much bias in width')                            

    # 배열 복사
    cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img

# -- RGB to GRAY
def convert_bgr2gray(data):
    # np.stack(배열_1, 배열_2, axis=0): 지정한 axis를 완전히 새로운 axis로 생각
    return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in data], axis=0)  # gray 변환
