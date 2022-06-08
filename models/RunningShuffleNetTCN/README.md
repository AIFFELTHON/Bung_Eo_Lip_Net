# RunningShuffleNetTCN

## Directory Trees

```
RunningShuffleNetTCN
├── ShuffleNetTCN
│   ├── configs
│   ├── data
│   ├── datasets
│   │   ├── audio_data
│   │   └── visual_data
│   │       ├── {word}
│   │       │   ├── test
│   │       │   │   ├── {word_00001.npz}
│   │       │   │   └── ...
│   │       │   ├── train
│   │       │   └── val
│   │       └── ...
│   ├── doc
│   ├── labels
│   ├── landmarks
│   │   └── hangeul_landmarks
│   │       ├── {word}
│   │       │   ├── test
│   │       │   │   ├── {word_00001.npz}
│   │       │   │   └── ...
│   │       │   ├── train
│   │       │   └── val
│   │       └── ...
│   ├── lipreading
│   │   └── models
│   ├── models
│   └── preprocessing
├── fonts
└── hangeul
    ├── {word}
    │   ├── test
    │   │   ├── {word_00001.avi}
    │   │   ├── {word_00001.txt}
    │   │   └── ...
    │   ├── train
    │   └── val
    └── ...
```

## Setting

1. 환경
    - GCP Ubuntu 18.04.6 LTS, GPU NVIDIA Tesla V100
        - GCP Ubuntu 에서 한글 폰트 다운로드
            - sudo apt-get update
            - sudo apt-get upgrade -y
            - sudo apt-get install fonts-nanum
        - GCP Ubuntu 에서 설치되어 있는 폰트 확인
            - fc-list | grep -i nanum
        - 폴더 트리 출력 라이브러리 설치
            - sudo apt-get install tree
        - 폴더 트리 출력
            - tree -d ./{directory}
    - Anaconda 사용
    - python 3.7.13
    - CUDA Version: 11.4
2. 라이브러리 설치
    - 기본 환경 세팅
        - pip install -r requirements.txt
    - audio 전처리하기 위해 ffmpeg 설치 필요
        - pip install ffmpeg-python
    - 참고: LRW 데이터셋 face landmark 는 csv 파일로 제공되므로 라이브러리(dlib, face_alignment 등) 필요 없음
        - pip install cmake
        - pip install dlib==19.17.0
        - pip install face-alignment
    - pytorch image tensor 처리 라이브러리인 torchvision 설치
        - pip install torchvision==0.2.0
    - 학습 관리 툴 wandb 설치
        - pip install wandb
3. 데이터셋: [The Oxford-BBC Lip Reading in the Wild (LRW)](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)
    - 데이터셋 다운로드 받으려면 서약서 작성해서 이메일(rob.cooper@bbc.co.uk) 보내고 승인받은 비밀번호 입력해야 함
        - [데이터셋 서약서 작성 문서 다운로드 페이지](https://www.bbc.co.uk/rd/projects/lip-reading-datasets)
    - **조건 없이 다운받을 수 있는 샘플 데이터 1개(AFTERNOON.mp4, AFTERNOON.txt)를 복제해서 실행 도전 성공**
    - **우리가 사용할 한국어 데이터 1개 복제해서 실행 도전 성공**
    - 구축한 한국어 데이터셋으로 train, test 완료

## Execution

### Preprocessing

1. crop_mouth_from_video.py
    - 영상 -> 이미지 프레임에서 입술만 잘라냄 -> numpy 변환 -> datasets/visual_data 경로에 .npz 저장
    - **--conver-gray 필수! 제대로 수행 안 될 경우, 스크립트에서 직접 arument 에서 default=True 설정**

```bash
# CropMousth.sh
python preprocessing/crop_mouth_from_video.py \
--video-direc ../sample/ \
--landmark-direc ./landmarks/LRW_landmarks/ \
--filename-path ../sample/AFTERNOON_detected_face.csv \
--save-direc ./datasets/visual_data/ \
--mean-face ./preprocessing/20words_mean_face.npy \
--convert-gray \
```

2. extract_audio_from_video.py
    - 영상 -> 소리 추출 -> (Sampling 진행) -> numpy 변환 -> datasets/audio_data 경로에 .npz 저장

```bash
# ExtractAudio.sh
python preprocessing/extract_audio_from_video.py \
--video-direc ../sample/ \
--filename-path ../sample/AFTERNOON_detected_face.csv \
--save-direc ./datasets/audio_data/ \
```

### Train

#### 수정된 부분

1. main.py

```python
def main():
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"  # GPU 선택 코드 추가

    ...  # 중략
```

2. lipreading/dataset.py

```python
def pad_packed_collate(batch):
    
    batch = np.array(batch, dtype=object)  # list 라서 numpy 로 변경, 내부 요소 리스트 길이가 달라서 dytpe=object 설정하는 코드 추가
    
    ...  # 중략
    
    return data, lengths, labels
```

3. lipreading/dataloaders.py

```python
def get_data_loaders(args):
    
    ...  # 중략
    
    dset_loaders = {x: torch.utils.data.DataLoader(
                        dsets[x],
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=pad_packed_collate,
                        pin_memory=True,
                        num_workers=2,  # GCP core 4개의 절반 => 2로 설정한 코드로 변경 # num_workers=args.workers,
                        worker_init_fn=np.random.seed(1)) for x in ['train', 'val', 'test']}
    
    
    return dset_loaders
```


#### 실행 도전

1. Train a visual-only model
    - 데이터 디렉토리 경로 설정 주의

```bash
# TrainVisual.sh
python main.py \
--config-path ./configs/lrw_resnet18_mstcn.json \
--annonation-direc ../sample/ \
--data-dir ./datasets/visual_data
```

2. Train an audio-only model
    - 데이터 디렉토리 경로 설정 주의

```bash
# TrainAudio.sh
python main.py \
--modality raw_audio \
--config-path ./configs/lrw_resnet18_mstcn.json \
--annonation-direc ../sample/ \
--data-dir ./datasets/audio_data
```

### Test

1. Evaluate the visual-only performance (lipreading)

```bash
# TestVisual.sh
python main.py \
--config-path ./configs/lrw_resnet18_mstcn.json \
--model-path ./models/lrw_resnet18_mstcn.pth.tar \
--data-dir ./datasets/visual_data/ \
--test
```

2. Evaluate the audio-only performance

```bash
# TrainAudio.sh
python main.py \
--modality raw_audio \
--config-path ./configs/lrw_resnet18_mstcn.json \
--model-path ./models/lrw_resnet18_mstcn_audio.pth.tar \
--data-dir ./datasets/audio_data/ \
--test
```

### Extract Embeddings

```bash
# ExtractEmbeddings.sh
python main.py \
--extract-feats \
--data-dir ./datasets/visual_data/ \
--config-path ./configs/lrw_resnet18_mstcn.json \
--model-path ./models/lrw_resnet18_mstcn.pth.tar \
--mouth-patch-path ./datasets/visual_data/AFTERNOON/test/ \
--mouth-embedding-out-path ../sample/embeddings
```
