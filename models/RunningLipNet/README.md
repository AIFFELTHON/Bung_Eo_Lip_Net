
```
RunningLipNet
        ├── LipNet
        │   ├── LICENSE
        │   ├── common
        │   │   ├── dictionaries
        │   │   │   ├── big.txt
        │   │   │   ├── grid.txt
        │   │   │   └── grid2.txt
        │   │   └── predictors
        │   ├── evaluation
        │   │   ├── confusion.py
        │   │   ├── phonemes.txt
        │   │   ├── predict.py
        │   │   ├── predict_batch.py
        │   │   ├── saliency.py
        │   │   └── stats.py
        │   ├── lipnet
        │   │   ├── __init__.py
        │   │   ├── core
        │   │   │   ├── __init__.py
        │   │   │   ├── decoders.py
        │   │   │   ├── layers.py
        │   │   │   └── loss.py
        │   │   ├── helpers
        │   │   │   ├── __init__.py
        │   │   │   ├── list.py
        │   │   │   └── threadsafe.py
        │   │   ├── lipreading
        │   │   │   ├── __init__.py
        │   │   │   ├── aligns.py
        │   │   │   ├── callbacks.py
        │   │   │   ├── curriculums.py
        │   │   │   ├── generators.py
        │   │   │   ├── helpers.py
        │   │   │   ├── videos.py
        │   │   │   └── visualization.py
        │   │   ├── model.py
        │   │   ├── model2.py
        │   │   └── utils
        │   │       ├── __init__.py
        │   │       ├── spell.py
        │   │       └── wer.py
        │   ├── predict
        │   ├── scripts
        │   │   └── extract_mouth_batch.py
        │   ├── setup.py
        │   ├── tests
        │   │   └── curriculum_test.py
        │   ├── train
        │   └── training
        │       ├── prepare.py
        │       ├── s1
        │       │   ├── align
        │       │   │   └── example.align
        │       │   └── results
        │       │       └── 2022:06:06:15:57:39
        │       │           ├── e04.csv
        │       │           ├── stats.csv
        │       │           └── weights04.h5
        │       └── train.py
        └── README.md
```

  
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
* 비디오 파일을 처리하기 위해 Ubuntu에서 ffmpeg 설치  
``apt install ffmpeg``  

* 한글 자막을 출력하기 위한 한글 폰트 설치  
``apt-get install fonts-nanum``

## face_landmarks.dat 추가하기  
* shape_predictor_68_face_landmarks.dat를 다운로드  
* 아래의 위치에 넣습니다. 

``/LipNet/predictors/{shape_predictor_68_face_landmarks.dat}`` 

## dataset  
  
STT를 이용해서 데이터를 만들어줍니다.  
  
* video 파일을 traning에 넣고 ``/LipNet/scripts/extract_mouth_batch.py``를 실행하여 mouth crop된 image frame을 생성합니다.  

* train dataset는
`` /LipNet/training/{speaker}/dataset/train/{speaker}/[train videos]``  
* val dataset는  
``/LipNet/training/{speaker}/dataset/val/{speaker}/[val videos]``  
로 넣어줍니다.  
  
* 영상의 발화 sequence를 ``align``파일로 만들어줍니다.  
  * ``align``파일의 예제 형식도 첨부해 두었습니다.
  * 영상의 형식은 ``{align}.align``입니다.
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
