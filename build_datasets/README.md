# STT-DataPreprocessing


```bash

├── STT
│   ├── wavs
│   ├── auto.py
│   ├── downloadFileGCS.py
│   ├── getAudioes.py
│   ├── sendToGCS.py
│   ├── sttJson.py
│   └── videos.json
└── video
    ├── avi
    ├── data
    │   └── [video_name]
    │       ├── face_video
    │       ├── face_video_align
    │       ├── video
    │       └── WordJson.json
    ├── MostWord
    │   └── [mostword]
    │       ├── align
    │       ├── video
    │       └── videoA
    ├── cropMouth.py
    ├── cutVideo.py
    ├── dictWord.py
    ├── faceRecognitionFromVideo.py
    ├── frameMaching.py
    ├── getVideos.py
    ├── makeDir.py
    ├── takeWord.py
    └── videos.json
``` 


## TODO

- 구글 STT를 사용해서 영상을 어절별로 자르는 것이 목표

### STT
- [x] 영상다운받기
- [x] 영상 GCS에 올리기
- [x] 영상 JSON 형태로 STT 추출

### Video
- [x] 영상 단어(어절)에 맞추어 자르기
- [x] 영상 얼굴 랜드마크 찾고 얼굴 자르기
- [x] 빈도수 높은 단어들만 모아서 따로 폴더에 담기

```

## 사용된 Library

```py
backports.zoneinfo==0.2.1
beautifulsoup4==4.11.1
cachetools==5.1.0
certifi==2022.5.18.1
charset-normalizer==2.0.12
dbus-python==1.2.16
decorator==4.4.2
distro-info==1.0
dlib==19.24.0
docopt==0.6.2
ffmpeg-python==0.2.0
future==0.18.2
google-api-core==2.8.0
google-auth==2.6.6
google-cloud-core==2.3.0
google-cloud-speech==2.14.0
google-cloud-storage==2.3.0
google-crc32c==1.3.0
google-resumable-media==2.3.3
googleapis-common-protos==1.56.1
grpcio==1.46.3
grpcio-status==1.46.3
idna==3.3
imageio==2.19.2
imageio-ffmpeg==0.4.7
Js2Py==0.71
moviepy==1.0.3
numpy==1.22.4
opencv-python==4.5.5.64
packaging==21.3
Pillow==9.1.1
pipwin==0.5.2
proglog==0.1.10
proto-plus==1.20.4
protobuf==3.19.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pydub==0.25.1
pyjsparser==2.7.1
pyparsing==3.0.9
PyPrind==2.11.3
pySmartDL==1.3.4
python-apt==2.2.1
pytube==12.1.0
pytz-deprecation-shim==0.1.0.post0
requests==2.27.1
rsa==4.8
scikit-video==1.1.11
scipy==1.8.1
six==1.16.0
sk-video==1.1.10
soupsieve==2.3.2.post1
SpeechRecognition==3.8.1
tqdm==4.64.0
tzdata==2022.1
tzlocal==4.2
unattended-upgrades==0.1
urllib3==1.26.9


```