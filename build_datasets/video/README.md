# video

- [x] 영상 단어(어절)에 맞추어 자르기 `cut
video`
- 
- [x] 어절단위로 자른 영상 정답라벨을 json파일에 저장하기 `data/{}/WordJson.json`
	- 파일명과 라벨 확인용도
```json
    {
        "VideoName": "ytn_NN_1",
        "words": [
            {
                "fileName": "ytn_NN_1_0_01",
                "word": "여러분",
                "duration": 0.7
            },
            {
                "fileName": "ytn_NN_1_0_03",
                "word": "뉴스가있는저녁",
                "duration": 0.9000000000000001
            },
```
- [x] align만들기 `data/{}/align`
 - word, duration, time
```
word: 여러분 
duration: 1.2 
starttime: 0.5 
endtime: 1.7 
```
- [x] 데이터고르기
	- [x] 너무 짧고 긴 영상 삭제 
	- `0.5에서 1.3초 사이 영상만 ` 
	- [x] 한명이 말하고 있지 않은 영상 삭제 
	- `dlib`이용해서 얼굴 박스 찾기
	- [x] 필요없는 단어 삭제 -> 빈도수 높은 단어만으로 데이터셋 구성하기 
	- word 모아놓은 txt파일 `wordlist.txt`로 빈도수 높은 단어 파악하기
		- `Visualization_of Korean_words_frequency.ipynb`


## 디렉토리 구조
```
video
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
```
1. 한글명으로 폴더/파일 생성은 작업에 결함이 생길 수 있음
2. 그래서 순서대로 숫자를 지정해줌
3. 예시) 영상 제목= {}

|data 폴더||
|:---|:---|
|video|data/{}/*video*/{}_1_001.avi||
|face_video|data/{}/*face_video*/{}_1_001.avi|
|align|data/{}/*align*/{}_1_001.txt|


## 준비물
1. `youtube` 영상 링크를 모아둔 `Json` 파일

```json
// -ex

{
	"파일이름":"링크",
	"002":"https://www.youtube.com/watch?v=bQF4umz6YCE&t=11s"

}
```
2. STT/wavs안에 `STT json`파일


## 사용법

```py

python3 cutVideo.py
python3 cropMouth.py
python3 faceRecognitionFromVideo.py
python3 takeWord.py
```

- `videos.json`에 영상 목록을 담고 `wav/` 폴더안에 STT json파일이 있어야함(`auto.py` 실행 후)
- `cutVideo`를 실행시키면 영상을 다운받고 타임스탬프대로 잘라줌 
	- (`avi/` 폴더안에는 `영상.avi`가 담김)
	- (`data/` 폴더 안에 `video/{}__.avi`, `align/{}__.txt`, `WordJson.json` 생성)
- `cropMouth`는 입술을 crop한 이미지를 프레임별로 잘라줌 I/O video/image (사용x)
- `faceRecognitionFromVideo`는 얼굴 box 검출 후 영상을 잘라줌 I/O video/video
	- (`data/` 폴더 안에 `face_video/{}__.avi` 생성)
- `takeWord`는 `MostWord{}/` 폴더 안에 빈도수 높은 단어들을 담기
	- `takeWord` 실행시 인자를 입력 받아서 지정한 이름의 `MostWord{}` 디렉토리가 생성됨
