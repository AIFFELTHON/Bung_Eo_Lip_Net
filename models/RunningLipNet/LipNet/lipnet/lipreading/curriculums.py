import numpy as np
from lipnet.lipreading.videos import VideoAugmenter

# 커리큘럼 클래스
class Curriculum(object):
    def __init__(self, rules):
        self.rules = rules
        self.epoch = -1

    # 커리큘럼 규칙 업데이트
    def update(self, epoch, train=True):
        self.epoch = epoch  # 에폭 설정
        self.train = train  # 학습여부
        current_rule = self.rules(self.epoch)  # 커리큐럼 규칙 설정
        self.sentence_length = current_rule.get('sentence_length') or -1  # 문장 길이 설정
        self.flip_probability = current_rule.get('flip_probability') or 0.0  # flip 확률 설정
        self.jitter_probability = current_rule.get('jitter_probability') or 0.0  # jitter 확률 설정

    # 커리큘럼 적용
    def apply(self, video, align):
        original_video = video  # 비디오 설정

        # 문장 길이 > 0 이라면
        if self.sentence_length > 0:
            # augment 적용한 비디오에 있는 subsentence 부분 video, align 으로 설정
            video, align = VideoAugmenter.pick_subsentence(video, align, self.sentence_length)

        # Only apply horizontal flip and temporal jitter on training
        # 학습 여부 True 라면
        if self.train:
            # 랜덤값 < flip 확률 이라면
            if np.random.ranf() < self.flip_probability:
                video = VideoAugmenter.horizontal_flip(video)  # 비디오 수평 뒤집기
            
            # 비디오에 jitter 확률 대로 temporal_jitter 적용
            video = VideoAugmenter.temporal_jitter(video, self.jitter_probability)
        video_unpadded_length = video.length  # 비디오 원본 길이
        video = VideoAugmenter.pad(video, original_video.length)  # 비디오 패딩 적용
        return video, align, video_unpadded_length  # augment 적용 비디오, align, 비디오 원본 길이 반환

    # 커리큘럼 출력
    def __str__(self):
        # 클래스명(학습여부, 문장길이, flip 확률, jitter 확률)
        return "{}(train: {}, sentence_length: {}, flip_probability: {}, jitter_probability: {})"\
            .format(self.__class__.__name__, self.train, self.sentence_length, self.flip_probability, self.jitter_probability)