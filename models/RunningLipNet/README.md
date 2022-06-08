

## 개발환경
* python 3.6
* Keras 2.0+
* Tensorflow 1.0+
* Ubuntu 16.04  
*  cuda 8.0  
*  cudnn 5
  

## 필수 패키지 설치
```
cd LipNet/
pip install -e .
```
비디오 파일을 처리하기 위해 Ubuntu에서 ffmpeg 설치  

``apt install ffmpeg``

## face_landmarks.dat 추가하기  
``/LipNet/predictors/{shape_predictor_68_face_landmarks.dat}`` 
shape_predictor_68_face_landmarks.dat를 다운로드 받아 해당 위치에 넣어주어야 합니다.
## dataset  
  
video 파일을 traning에 넣고 ``/LipNet/scripts/extract_mouth_batch.py``를 실행하여 mouth crop된 image frame을 생성합니다.  

train dataset는
`` /LipNet/training/{speaker}/dataset/train/{speaker}/[train videos]``  
val dataset는  
``/LipNet/training/{speaker}/dataset/val/{speaker}/[val videos]``  
로 넣어줍니다.  
  
영상의 발화 sequence를 ``align``파일로 만들어줍니다.  
``align``파일의 예제 형식도 첨부해 두었습니다.
영상의 형식은 ``{align}.align``입니다.
``/LipNet/training/{speaker}/dataset/{speaker}/align/[aligns]``에 넣어줍니다.
  
  
## train  
```
python3 training/train.py {speaker}
```

## predict  
```
./predict [path to weight] [path to video]
```


## License
MIT License
