# video

- [x] 영상 단어(어절)에 맞추어 자르기
- [x] 입술 크롭한 데이터 폴더명 지정하기
- [x] 어절단위의 영상 정답라벨을 json파일에 저장하기
- [x] align

- [ ] 크롭하고나서 생각해 볼것! 데이터고르기
	- [x] 너무 짧고 긴 영상 삭제 -> time 기준? word 기준?
	- [x] 한명이 말하고 있지 않은 영상 삭제 ->입술을 찾지 못할 때
	- [x] 필요없는 단어 삭제 -> 빈도수 높은 단어만으로 데이터셋 구성하기

- [x] 영상 얼굴 랜드마크 찾고 얼굴 자르기



## 사용법

```py

python3 cutVideo.py
python3 cropMouth.py
python3 faceRecognitionFromVideo.py
python3 takeWord.py
```

## 디렉토리 구조
1. 한글명으로 폴더/파일 생성은 작업에 결함이 생길 수 있음
2. 그래서 순서대로 숫자를 지정해줌
3. 예시) 영상 제목= 001

|폴더명|Video|Image|
|:---|:---|:---|
|변경전|data/001/001_1/*video*/001_001안녕하세요.avi|data/001/001_1/*image*/001_001안녕하세요/1.png|
|변경후|data/001/001_1/*video*/001_001.avi|data/001/001_1/*image*/001_1_001/1.png|
|변경후후|data/001/*video*/001_1_001.avi|data/001/*image*/001_1_001/1.png|



## 준비물
- `youtube` 영상 링크를 모아둔 `Json` 파일

```json
// -ex

{
	"파일이름":"링크",
	"002":"https://www.youtube.com/watch?v=bQF4umz6YCE&t=11s"

}
```

