# STT

## TODO

- [x] 영상다운받기 `getAudioes`
- [x] 영상 GCS에 올리기 `sendToGCS`
- [x] 영상 JSON 형태로 STT 추출 `sttJson`
- [x] json파일 경로 './STT/wavs/'
- [x] GCS Timeout 에러 
	- wav 영상길이가 13분을 넘어갈 갈경우에 STT가 반환되지않음
	- 영상은 13분을 넘지않은 영상만 찾거나 10분 이하로 잘라서 쓰는게 좋음 


## 준비물
- key.json
	- GCP 계정 key
- videos.json
	- `youtube` 영상 링크를 모아둔 `Json` 파일

```json
// -ex

{
	"파일이름":"링크",
	"002":"https://www.youtube.com/"

}
```

## 사용법

```py
python3 auto.py
```

- ```videos.json```에 다운받을 영상 목록을 작성한후 ```auto.py```를 실행
- STT json파일은 ```wavs/```폴더 안에 담김
