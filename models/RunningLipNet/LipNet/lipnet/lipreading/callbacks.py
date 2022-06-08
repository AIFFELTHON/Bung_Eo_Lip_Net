from lipnet.utils.wer import wer_sentence
from nltk.translate import bleu_score
import numpy as np
import editdistance
import keras
import csv
import os

# 통계 클래스
class Statistics(keras.callbacks.Callback):

    def __init__(self, model_container, generator, decoder, num_samples_stats=256, output_dir=None):
        self.model_container = model_container
        self.output_dir = output_dir
        self.generator = generator
        self.num_samples_stats = num_samples_stats
        self.decoder = decoder

        # 디렉토리가 비어있지 않고 디렉토리가 존재하지 않는다면
        if output_dir is not None and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)  # 디렉토리 생성

    # 통계값 가져오기
    def get_statistics(self, num):
        num_left = num
        data = []

        while num_left > 0:
            output_batch    = next(self.generator)[0]  # 제너레이터 배치
            num_proc        = min(output_batch['the_input'].shape[0], num_left)  # 번호
            y_pred          = self.model_container.predict(output_batch['the_input'][0:num_proc])  # 추론 결과
            input_length    = output_batch['input_length'][0:num_proc]  # 인풋 길이
            decoded_res     = self.decoder.decode(y_pred, input_length)  # 디코딩결과

            for j in range(0, num_proc):
                data.append((decoded_res[j], output_batch['source_str'][j]))

            num_left -= num_proc

        mean_cer, mean_cer_norm    = self.get_mean_character_error_rate(data)  # 평균 cer 계산
        mean_wer, mean_wer_norm    = self.get_mean_word_error_rate(data)  # 평균 wer 계산
        mean_bleu, mean_bleu_norm  = self.get_mean_bleu_score(data)  # 평균 bleu 계산

        return {
            'samples': num,  # sample num
            'cer': (mean_cer, mean_cer_norm),  # mean character error rate
            'wer': (mean_wer, mean_wer_norm),  # mean word error rate
            'bleu': (mean_bleu, mean_bleu_norm)  # mean bleu
        }  # 딕셔너리 반환

    # 종합/정규화종합 평균 계산
    def get_mean_tuples(self, data, individual_length, func):
        total       = 0.0  # 종합
        total_norm  = 0.0  # 종합 정규화
        length      = len(data)  # 데이터 길이
        for i in range(0, length):
            val         = float(func(data[i][0], data[i][1]))
            total      += val
            total_norm += val / individual_length
        return (total/length, total_norm/length)  # 평균

    # 평균 cer
    def get_mean_character_error_rate(self, data):
        mean_individual_length = np.mean([len(pair[1]) for pair in data])
        return self.get_mean_tuples(data, mean_individual_length, editdistance.eval)

    # 평균 wer
    def get_mean_word_error_rate(self, data):
        mean_individual_length = np.mean([len(pair[1].split()) for pair in data])
        return self.get_mean_tuples(data, mean_individual_length, wer_sentence)

    # 평균 bleu
    def get_mean_bleu_score(self, data):
        wrapped_data = [([reference],hypothesis) for reference,hypothesis in data]
        return self.get_mean_tuples(wrapped_data, 1.0, bleu_score.sentence_bleu)

    # epoch, 평균 통계값 기록
    def on_train_begin(self, logs={}):
        # write binary mode 로 csv 파일 작성
        with open(os.path.join(self.output_dir, 'stats.csv'), 'w', encoding='utf-8', newline='') as csvfile:
            csvw = csv.writer(csvfile)
            csvw.writerow(["Epoch", "Samples", "Mean CER", "Mean CER (Norm)", "Mean WER", "Mean WER (Norm)", "Mean BLEU", "Mean BLEU (Norm)"])

    # epoch, 통계값 기록
    def on_epoch_end(self, epoch, logs={}):
        stats = self.get_statistics(self.num_samples_stats)  # 통계값 가져오기

        # 통계값 출력
        print('\n\n[Epoch %d] Out of %d samples: [CER: %.3f - %.3f] [WER: %.3f - %.3f] [BLEU: %.3f - %.3f]\n'
              % (epoch, stats['samples'], stats['cer'][0], stats['cer'][1], stats['wer'][0], stats['wer'][1], stats['bleu'][0], stats['bleu'][1]))

        # 디렉토리가 비어있지 않다면
        if self.output_dir is not None:
            # write binary mode 로 csv 파일 작성
            with open(os.path.join(self.output_dir, 'stats.csv'), 'w', encoding='utf-8', newline='') as csvfile:
                csvw = csv.writer(csvfile)
                csvw.writerow([epoch, stats['samples'],
                               "{0:.5f}".format(stats['cer'][0]), "{0:.5f}".format(stats['cer'][1]),
                               "{0:.5f}".format(stats['wer'][0]), "{0:.5f}".format(stats['wer'][1]),
                               "{0:.5f}".format(stats['bleu'][0]), "{0:.5f}".format(stats['bleu'][1])])


# 시각화 클래스
class Visualize(keras.callbacks.Callback):

    def __init__(self, output_dir, model_container, generator, decoder, num_display_sentences=10):
        self.model_container = model_container
        self.output_dir = output_dir
        self.generator = generator
        self.num_display_sentences = num_display_sentences
        self.decoder = decoder

        # 디렉토리가 존재하지 않는다면
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)  # 디렉토리 생성

    # 정답, 디코딩결과 기록
    def on_epoch_end(self, epoch, logs={}):
        output_batch = next(self.generator)[0]

        y_pred       = self.model_container.predict(output_batch['the_input'][0:self.num_display_sentences])  # 추론 결과
        input_length = output_batch['input_length'][0:self.num_display_sentences]  # 인풋 길이
        res          = self.decoder.decode(y_pred, input_length)  # 디코딩결과
        print(res)
        # write binary mode 로 csv 파일 작성
        with open(os.path.join(self.output_dir, 'e%02d.csv' % (epoch)), 'w', encoding='utf-8', newline='') as csvfile:
            csvw = csv.writer(csvfile)
            csvw.writerow(["Truth", "Decoded"])
            for i in range(self.num_display_sentences):
                csvw.writerow([(output_batch['source_str'][i]), res[i]])