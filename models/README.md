<h1 align="center">Lipreading Models</h1>

<div align="right">
  AIFFEL DAEGU 1 TEAM 뻐끔뻐끔
  <br>
  <i>#Tags: Lipreading, LipNet, ShuffleNet-TCN, CV, NLP</i>
</div>

---

## 📌 TABLE OF CONTENTS

- [I/O](#io)
- [공통 구조](#공통-구조)
- [LipNet (2016)](#lipnet-2016)
- [ShuffleNetTCN (2020)](#shufflenettcn-2020)

---

## I/O

![Lipreading Model Input Output][Lipreading Model Input Output]

[Lipreading Model Input Output]: imgs/Lipreading_input_and_output.png

- Input: Video(.avi), Align(.txt)
- Output: Text(.txt) → Video(.gif, .avi)

## 공통 구조

|LipNet|ShuffleNetTCN|
|:---:|:---:|
|![LipNet Architecture Description][LipNet Architecture Description]|![ShuffleNetTCN Architecture Description][ShuffleNetTCN Architecture Description]|

[LipNet Architecture Description]: imgs/LipNet_architecture_description.png
[ShuffleNetTCN Architecture Description]: imgs/ShuffleNetTCN_architecture_description.png

- 딥러닝 2단계 접근법
  - Frontend: 3D-CNN(3D conv layer + deep 2D conv)
  - Backent: LSTM, Attention Mechanisms, self-attention module, TCN (Temporal Convolutional Networks)

**[⬆ back to top](#-table-of-contents)**

---

## LipNet (2016)

### Key Contributions

1. 최초의 end-to-end 문장 단위 모델
2. GRID Corpus dataset

### Process

![LipNet Process][LipNet Process]

[LipNet Process]: imgs/LipNet_architecture.png

- STCNN: video 에서 시간의 흐름과 공간의 차원을 모두 convolution
- bi-LSTM: STCNN 의 output sequence 정보를 전파하기 위해 사용, 정보 흐름 제어 학습
- CTC Loss: Target sequence 와 output sequence 길이가 다를 때 사용
- Label → UNICODE (encoding) → 한국어 (decoding)

### 단점 파악 및 분석

- 연산량 이슈: GCP 환경에서 학습을 돌릴 때 GCP 가 꺼지는 현상 발생
- 원인 분석
  - 한국어 character 개수가 영어 character 에 비해 가짓수가 많음
  - 모델이 무거운 것(하이퍼 파라미터 많음)으로 파악됨
- 개선 방법
  - 예전 모델인 LipNet 말고 최신 모델을 찾아볼 것 → 최신 논문 찾아보기
  - 웬만하면 가벼운 모델일 것 → 논문에서 연산량 또는 학습 시간에 대한 내용 집중해서 보기
  - 성능이 좋을 것 → SOTA 찾아보기

### 결론

SOTA 를 달성한 가볍고 최신 단어 단위 모델인 ShuffleNetTCN 사용

**[⬆ back to top](#-table-of-contents)**

---

## ShuffleNetTCN (2020)

### Key Contributions

1. Backbone 교체
    - ResNet-18 → ShuffleNetV2
    - 파라미터 5배 감소, FLOPs 12배 감소
2. Shuffle Grouped Convolution
    - ShuffleNet 에서 제안
    - 주기적으로 그룹 간 채널을 섞어서 정보가 교환되도록 만든 Group Conv 방식
3. TCN (Temporal Convolution Network)
    - 1D conv 를 Sequence 데이터에 적용
    - 병렬 연산 → 직렬 RNN 보다 빠름
    - receptive field 크기 조절 가능
    - 하나의 layer 에 대하여 같은 파라미터 공유 → 메모리 소요 적음
4. DS-TCN (Depthwise Separable Temporal Convolution Network)
    - Depthwise Conv: 공간 특징 추출
    - Pointwise Conv: 채널 특징 추출
    - 표준 Conv 보다 연산량 8 ~ 9배 감소
    - 최종적으로 DS-MS-TCN 사용
5. Knowledge Distillation
    - 큰 모델(Teacher)로부터 증류한 지식 → 작은 모델(Student)로 전달
    - Teacher Network 와 Student Network 는 모두 같은 도메인 → Self-Distillation 과정

### Process

![ShuffleNetTCN Process][ShuffleNetTCN Process]

[ShuffleNetTCN Process]: imgs/ShuffleNetTCN_architecture.png

- 3D Conv: video 처리
- ResNet-18 or ShuffleNet: image frames 처리
- MS-TCN: text output 생성

**[⬆ back to top](#-table-of-contents)**
