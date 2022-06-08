<h1 align="center">AIFFELTHON</h1>

<div align="right">
  AIFFEL DAEGU 1 TEAM 뻐끔뻐끔
  <br>
  <i>#Tags: Lipreading, LipNet, ShuffleNet-TCN, CV, NLP</i>
</div>

---

## 📌 TABLE OF CONTENTS

- [📆 진행 기간](#-진행-기간-20220420--20220610)
- [💋 팀 소개: 뻐끔뻐끔](#-팀-소개-뻐끔뻐끔)
- [🐡 프로젝트명: 붕어립(Bung-Eo-Lip)](#-프로젝트-소개-붕어립bung-eo-lip)
- [🗂 한국어 데이터셋 구축](#-한국어-데이터셋-구축)
- [🌟 데이터 전처리](#-데이터-전처리)
- [🏭 모델 선정](#-모델-선정-shufflenet-tcn)
- [🤔 데모](#-데모)
- [👋 서비스 개발](#-서비스-개발)
- [🔗 참고자료](#-참고자료)

---

## 📆 진행 기간 (2022.04.20 ~ 2022.06.10)

|단계|내용|M1|M2|H1|H2|H3|H4|H5|H6|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|데이터셋 이해/제작|🟥|-|-|-|-|-|-|-|
|1|관련 논문 리뷰|🟥|-|-|-|-|-|-|-|
|1|모델 설정|-|🟥|🟥|🟥|-|-|-|-|
|1|검수|-|-|-|🟥|-|-|-|-|
|2|서비스 개발|-|-|-|-|🟦|🟦|🟦|-|
|2|검수|-|-|-|-|-|-|🟦|-|
|3|전체 유지보수|-|-|-|-|-|-|-|🟦|

- 🟥: Aiffelthon 기간동안 진행 완료
- 🟦: KDT Hackerthon 2022 기간에 이어서 마무리

**[⬆ back to top](#-table-of-contents)**

---

## 💋 팀 소개: 뻐끔뻐끔

### 역할

|번호|조원|깃허브|역할|
|:---:|:---:|:---:|---|
|1|박혜령|[➡️][HRPzz]|팀장, 논문 리뷰, **모델링**, 총괄|
|2|성은지|[➡️][eunji1]|팀원, 논문 리뷰, **데이터셋 구축**|
|3|이창수|[➡️][imfreeman1]|팀원, 논문 리뷰, 모델링, **개발 환경 구축**|
|4|김선아|[➡️][Seona056]|팀원, **논문 리뷰**, 모델링, 발표|
|5|이동섭|[➡️][xddf]|팀원, 논문 리뷰, 데이터셋 구축 보조, **도메인 조사**|

[HRPzz]: https://github.com/HRPzz
[eunji1]: https://github.com/eunji1
[imfreeman1]: https://github.com/imfreeman1
[Seona056]: https://github.com/Seona056
[xddf]: https://github.com/xddf

![Total Process][Total Process]

[Total Process]: imgs/Total_process.png

### 협업 툴 사용

|번호|항목|링크|용도|
|:---:|---|:---:|---|
|1|Gather Town|-|만남, 소통|
|2|Figma|[🔗][Figma]|자료 정리|
|3|Google Shared Drive|-|파일 공유|
|4|Colab Jupyter Notebook|-|코드 및 마크다운 작성|
|5|GitHub|[🔗][GitHub]|업로드 및 제출|
|6|Notion|[🔗][Notion]|프로젝트 진행 기록|
|7|Google Shared PPT|[🔗][PPT]|발표 PPT 작성|
|8|GCP|-|개발 환경 구축|

[Figma]: https://www.figma.com/file/yADblEOzjSj2jo43xdCHn4/AIFFELTHON?node-id=0%3A1
[Google Shared Drive]: https://drive.google.com/drive/folders/1WlgzdIVu4ZOR0R1_RS2aa8YXVIvQPqn0?usp=sharing
[Colab Jupyter Notebook]: https://colab.research.google.com/drive/1UY0x-3ggFeSodhk6rokyZJDIl5swzLZO
[GitHub]: Bung_Eo_Lip_Net.ipynb
[Notion]: https://www.notion-pinotnoir056.com/8f71053c-b272-49dc-ac72-1463c747f382
[PPT]: https://docs.google.com/presentation/d/1ZKAedBvUDkMbuSBS6xNLcyWJgRcmYxvd/edit?usp=sharing&ouid=109572844521289026032&rtpof=true&sd=true

**[⬆ back to top](#-table-of-contents)**

---

## 🐡 프로젝트 소개: 붕어립(Bung-Eo-Lip)

||붕어립(Bung-Eo-Lip) 프로젝트란 ❓|
|:---:|---|
|![Intro IMG][Intro IMG]|- 영상의 입 모양 모션을 탐지하여 자막 서비스를 할 수 있는 프로젝트이다.<br>- 최근 미국과 영국에서 늘어나고 있는 배리어 프리(barrier-free) 서비스 수준의 결과물을 내는 것이 목표이다.<br>&emsp;- 배리어 프리란? 장애인을 포함한 모든 사회의 구성원이 살기 좋은 사회를 만들기 위해 물리적·제도적·심리적 장벽을 허물자는 운동이다.|

[Intro IMG]: https://user-images.githubusercontent.com/44178037/165084111-c31f42d1-2680-4490-9b30-f10639878398.png

- 연구 목표: 한국어 독순술에 대한 AI 서비스
  - [LV1] 입 모양 인식 모델 구현
  - [LV2] 서비스 개발
  - [LV3] 발표 준비

**[⬆ back to top](#-table-of-contents)**

---

## 🗂 한국어 데이터셋 구축

### 전체 프로세스 진행 파이프라인

![Data Collection Pipeline][Data Collection Pipeline]

[Data Collection Pipeline]: imgs/Korean_data_collection_pipeline.png

### 데이터셋 구축 파이프라인 - Google STT 자동화

![Korean Datasets][Korean Datasets]

[Korean Datasets]: imgs/Google_STT_modularization.png

### 데이터셋 구축 순서

1. 아나운서 유튜브 영상(.avi) 다운로드
2. 음원(.wav) 채널 모노 변경
3. Google STT(Speech-To-Text) API 사용 → Timestamp 찍힌 파일(.json) 생성
4. Timestamp(startTime, endTime) 대로 단어/어절 영상(.avi) 잘라내기
5. model input 에 맞는 align 파일(.txt) 생성
6. 잘라낸 단어/어절 영상(.avi) 검수: align 에 적힌 단어/어절 텍스트와 맞는지 영상의 소리 데이터로 loss 확인
7. 빈도수가 높은 단어/어절 데이터만 따로 폴더링해서 최종 사용

**[⬆ back to top](#-table-of-contents)**

---

## 🌟 데이터 전처리

### 전처리 요약

![Preprocessing][Preprocessing]

[Preprocessing]: imgs/Preprocessing.png

### 전처리 순서

1. 단어/어절 영상 입력
2. 이미지 프레임 추출
3. 프레임 개수(e.g. 29개) 맞추기
4. 각 프레임마다 GrayScale 변환, Face Landmark 찾기, 입술 Crop
5. .npz 파일 저장

**[⬆ back to top](#-table-of-contents)**

---

## 🏭 모델 선정: ShuffleNet-TCN

|LipNet|ShuffleNet-TCN|
|:---:|:---:|
|![LipNet][LipNet]|![ShuffleNet-TCN][ShuffleNet-TCN]|

[LipNet]: imgs/LipNet_architecture.png
[ShuffleNet-TCN]: imgs/ShuffleNet_TCN_architecture.png

1. LipNet 으로 진행하다가 연산량 이슈로 학습이 더 이상 불가능한 상황 발생
2. 경량화에 초점을 맞춘 ShuffleNet-TCN 모델로 변경
3. 앞서 구축해놓은 데이터셋으로 Train, Test 진행
5. 데모 실행

**[⬆ back to top](#-table-of-contents)**

---
## 🤔 데모

### 데모 실행

|Input|Process|Output|
|:---:|:---:|:---:|
|![Input][Input]|![Process][Process]|![Output][Output]|

[Input]: imgs/Demo_origin_GIF.gif
[Process]: imgs/Demo_process.png
[Output]: imgs/Demo_predict_GIF.gif

- Input: '함께' 단어 영상
- Process
  - ShuffleNet-TCN 으로 학습시킨 Pre-trained 파일(100 epochs) 불러와서 진행
  - 1.x초 영상 처리 시간 → 대략 15초
- Output: '그리고' 텍스트, 자막붙은 영상


### 데모 분석 - Bad Case Analysis

|번호|결과 원인 분석|개선 방법|
|:---:|---|---|
|1|학습에 사용한 데이터가 적어서 데모 output 이 별로인 것으로 판단됨|학습에 사용할 데이터셋을 더 많이 늘려서 train, test 시도|
|2|Google STT API 가 불완전하기 때문에 구축한 영상 데이터의 Loss 존재|음성 데이터와 대조하여 영상 데이터의 Loss 를 줄여서 구축|
|3|하이퍼 파라미터의 최적값을 찾아내지 못함|모델 학습에 사용한 하이퍼 파라미터 변경 및 학습 횟수 변경을 통해 개선|

**[⬆ back to top](#-table-of-contents)**

---

## 👋 서비스 개발

### KDT Hackerthon 2022 기간에 이어서 마무리

|서비스 구현 프로세스|설명|
|:---:|---|
|![Service][Service]|1. BentoML 을 통해 Backend 와 AI 모델 매핑<br>2. ReactJS 를 통해 Frontend 구축<br>3. Web Domain 주소 설정<br>4. Web Service 배포|

[Service]: imgs/Service_process.png

**[⬆ back to top](#-table-of-contents)**

---

## 🔗 참고자료

### Lipreading 논문 도표

![Paper chart 1][Paper chart 1]

[Paper chart 1]: imgs/Paper_chart_1.png

### 차용한 논문 3가지

![Paper chart 2][Paper chart 2]

[Paper chart 2]: imgs/Paper_chart_2.png

### 차용한 논문 관련 링크

|번호|논문 제목|년도|논문 링크|깃허브|
|:---:|---|:---:|:---:|:---:|
|1|LipNet: End-to-End Sentence-level Lipreading|2016|[📋][LipNet Paper]|[📁][LipNet GitHub 1], [📁][LipNet GitHub 2]|
|2|Towards Practical Lipreading with Distilled and Efficient Models|2020|[📋][ShuffleNet-TCN Paper]|[📂][ShuffleNet-TCN GitHub]|
|3|LRWR: Large-Scale Benchmark for Lip Reading in Russian language|2021|[📋][LRWR Paper]|-|

[LipNet Paper]: https://arxiv.org/abs/1611.01599
[LipNet GitHub 1]: https://github.com/rizkiarm/LipNet
[LipNet GitHub 2]: https://github.com/Fengdalu/LipNet-PyTorch

[ShuffleNet-TCN Paper]: https://arxiv.org/abs/2007.06504
[ShuffleNet-TCN GitHub]: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks

[LRWR Paper]: https://arxiv.org/abs/2109.06692

**[⬆ back to top](#-table-of-contents)**
