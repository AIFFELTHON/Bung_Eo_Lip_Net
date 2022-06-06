# STT

## TODO

- [x] 영상다운받기
- [x] 영상 GCS에 올리기
- [x] 영상 JSON 형태로 STT 추출
- [x] 503에러 해결
	- [x] json파일 경로 './STT/wavs/'
- [x] Timeout 에러
	- [x] 영상길이가 너무 길경우에 STT변환 x (약 13분 이상)


## 준비물
- videos.json
- `youtube` 영상 링크를 모아둔 `Json` 파일

```json
// -ex

{
	"파일이름":"링크",
	"002":"https://www.youtube.com/watch?v=bQF4umz6YCE&t=11s"

}
```

## 사용법

```py
python3 auto.py

```