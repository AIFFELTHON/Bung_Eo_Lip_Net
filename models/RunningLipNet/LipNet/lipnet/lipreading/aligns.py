# -*- coding: utf-8 -*-
import numpy as np

class Align(object):
    def __init__(self, absolute_max_string_len=32, label_func=None):
        self.label_func = label_func    #label_fuction
        self.absolute_max_string_len = absolute_max_string_len  #완전한 문장의 길이

    def from_file(self, path):
        with open(path, 'r') as f: # path를 'read'모드로 열기
            lines = f.readlines()   # readlines()로 path내용 읽기.
            # print(f.readlines())
        # print('==========================here====================')
        align = [(int(y[0])/1000, int(y[1])/1000, y[2]) for y in [x.strip().split(" ") for x in lines]] # align
        self.build(align)
        return self

    def from_array(self, align): # 
        self.build(align)
        return self

    def build(self, align): # 아래 기능들을 build 함.
        self.align = self.strip(align, ['sp','sil'])
        # print(self.align)
        self.sentence = self.get_sentence(align)
        # print(self.sentence)
        self.label = self.get_label(self.sentence)
        # print(self.label)
        self.padded_label = self.get_padded_label(self.label)
        # print(self.padded_label)

    def strip(self, align, items):  # sub[2]가 items에 없을때 sub를 return sub[2]가 단어str로 추정
        return [sub for sub in align if sub[2] not in items]

    def get_sentence(self, align):  # 'sp'랑 sil이 아닌 단어들을 join
        return " ".join([y[-1] for y in align if y[-1] not in ['sp', 'sil']])

    def get_label(self, sentence): # input으로 들어온 label_func를 sentence에 적용.
        return self.label_func(sentence)

    def get_padded_label(self, label):
        padding = np.ones((self.absolute_max_string_len-len(label))) * -1 # -1 array를 만든다.(크기는 max_stringlen-len(label))
        return np.concatenate((np.array(label), padding), axis=0)

    @property
    def word_length(self): # ' '로 split해서 길이를 return
        return len(self.sentence.split(" "))

    @property
    def sentence_length(self): # sentence의 길이 return
        return len(self.sentence)

    @property
    def label_length(self): # label의 길이 return
        return len(self.label)