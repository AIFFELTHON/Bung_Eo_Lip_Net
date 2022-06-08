#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


"""Transforms mp4 audio to npz. Code has strong assumptions on the dataset organization!"""

import os
import librosa  # 음원 데이터 분석 라이브러리
import argparse  # 명령행 인자를 파싱해주는 모듈

from utils import *  # utils.py 모듈에 있는 모든 함수(read_txt_lines(), save2npz(), read_video()) 불러오기


# 인자값을 받아서 처리하는 함수
def load_args(default_config=None):
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='Extract Audio Waveforms')
    
    # 입력받을 인자값 등록
    # -- utils
    parser.add_argument('--video-direc', default=None, help='raw video directory')
    parser.add_argument('--filename-path', default='./lrw500_detected_face.csv', help='list of detected video and its subject ID')
    parser.add_argument('--save-direc', default=None, help='the directory of saving audio waveforms (.npz)')
    # -- test set only
    parser.add_argument('--testset-only', default=False, action='store_true', help='process testing set only')

    # 입력받은 인자값을 args에 저장 (type: namespace)
    args = parser.parse_args()
    return args

args = load_args()  # args 파싱 및 로드

lines = open(args.filename_path).read().splitlines()  # 문자열을 '\m' 기준으로 쪼갠 후 list 생성
lines = list(filter(lambda x: 'test' == x.split('/')[-2], lines)) if args.testset_only else lines   # args.testset_only 값이 있다면 test 폴더 속 파일명만 불러와서 list 생성, 아니라면 원래 lines 그대로 값 유지

# lines 개수만큼 반복문 실행
for filename_idx, line in enumerate(lines):

    # 파일명, 사람id
    filename, person_id = line.split(',')
    print('idx: {} \tProcessing.\t{}'.format(filename_idx, filename))  # 파일 인덱스번호, 파일명 출력

    video_pathname = os.path.join(args.video_direc, filename+'.mp4')  # 영상디렉토리 + 파일명.mp4 
    dst_pathname = os.path.join( args.save_direc, filename+'.npz')  # 저장디렉토리 + 파일명.npz

    # 파일이 있는지 확인, 없으면 AssertionError 메시지를 띄움
    assert os.path.isfile(video_pathname), "File does not exist. Path input: {}".format(video_pathname)  # 원하는 조건의 변수값을 보증하기 위해 사용

    # wav 파일 읽는 라이브러리: librosa
    # librosa 로 데이터를 읽으면 데이터 범위가 [-1,1]로 정규화됨
    # librosa 입력에서 sr=None 으로 지정하지 않고 임의의 sample_rate를 설정하면 load할 때 resampling 수행함
    data = librosa.load(video_pathname, sr=16000)[0][-19456:]
    save2npz(dst_pathname, data=data)  # librosa 로 읽은 데이터를 npz 형식으로 저장