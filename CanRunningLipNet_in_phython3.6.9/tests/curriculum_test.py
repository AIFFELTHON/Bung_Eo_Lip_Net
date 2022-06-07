from lipnet.lipreading.curriculums import Curriculum
from lipnet.lipreading.videos import Video
from lipnet.lipreading.aligns import Align
from lipnet.lipreading.helpers import text_to_labels, labels_to_text
from lipnet.lipreading.visualization import show_video_subtitle
import numpy as np

# 규칙 설정
def rules(epoch):
    # 문장 길이, flip 확률, jitter 확률 설정
    if epoch == 0:
        return {'sentence_length': 1, 'flip_probability': 0, 'jitter_probability': 0}
    if epoch == 1:
        return {'sentence_length': 2, 'flip_probability': 0.5, 'jitter_probability': 0}
    if epoch == 2:
        return {'sentence_length': 3, 'flip_probability': 0.5, 'jitter_probability': 0.05}
    if epoch == 3:
        return {'sentence_length': -1, 'flip_probability': 0, 'jitter_probability': 0}
    if epoch == 4:
        return {'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0}
    return {'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0.05}  # 딕셔너리 반환

# 결과 출력
# (커리큘럼 적용 video, 커리큘럼 적용 align, 원본 video, 원본 align)
def show_results(_video, _align, video, align):
    # 자막 추가한 영상(이미지 프레임) 보여주기
    show_video_subtitle(frames=_video.face, subtitle=_align.sentence)

    print ("Video: ")
    print (_video.length)  # 커리큘럼 적용된 영상 길이
    print (np.array_equiv(_video.mouth, video.mouth),)  # 원본과 커리큘럼 적용된 영상 입 검출 비교
    print (np.array_equiv(_video.data, video.data),)  # 원본과 커리큘럼 적용된 영상 데이터 비교
    print (np.array_equiv(_video.face, video.face))  # 원본과 커리큘럼 적용된 영상 얼굴 검출 비교

    print ("Align: ")
    print (labels_to_text(_align.padded_label.astype(np.int)))
    print (_align.padded_label)  # 커리큘럼 적용된 align
    print (_align.label_length)  # 커리큘럼 적용된 align 의 라벨 길이
    print (np.array_equiv(_align.sentence, align.sentence),)  # 원본과 커리큘럼 적용된 align 문장 비교
    print (np.array_equiv(_align.label, align.label),)  # 원본과 커리큘럼 적용된 라벨 비교
    print (np.array_equiv(_align.padded_label, align.padded_label))  # 원본과 커리큘럼 적용된 패딩라벨 비교

curriculum = Curriculum(rules)  # 커리큘럼 설정

video = Video(vtype='face', face_predictor_path='evaluation/models/shape_predictor_68_face_landmarks.dat')  # dlib 으로 face landmark 찾기
video.from_video('evaluation/samples/id2_vcd_swwp2s.mpg')  # swwp2s 영상 가져오기

# swwp2s 영상의 align 가져오기
align = Align(absolute_max_string_len=32, label_func=text_to_labels).from_file('evaluation/samples/swwp2s.align')

print ("=== TRAINING ===")
for i in range(6):
    curriculum.update(i, train=True)  # 커리큘럼 업데이트, 학습 여부 True
    print (curriculum)  # 커리큘럼 출력
    _video, _align, _ = curriculum.apply(video, align)  # 커리큘럼 적용된 video, align 설정
    show_results(_video, _align, video, align)  # 결과 출력

print ("=== VALIDATION/TEST ===")
for i in range(6):
    curriculum.update(i, train=False)  # 커리큘럼 업데이트, 학습 여부 False
    print (curriculum)  # 커리큘럼 출력
    _video, _align, _ = curriculum.apply(video, align)  # 커리큘럼 적용된 video, align 설정
    show_results(_video, _align, video, align)  # 결과 출력