import os
import glob
import torch
import random
import librosa
import numpy as np
import sys
from lipreading.utils import read_txt_lines


# dataloaders.py에서 사용된 MyDataset
# dsets = {partition: MyDataset(
#                 modality=args.modality,
#                 data_partition=partition,
#                 data_dir=args.data_dir,
#                 label_fp=args.label_path,
#                 annonation_direc=args.annonation_direc,
#                 preprocessing_func=preprocessing[partition],
#                 data_suffix='.npz'
#                 ) for partition in ['train', 'val', 'test']}


class MyDataset(object):

    def __init__(self, modality, data_partition, data_dir, label_fp, annonation_direc=None,
        preprocessing_func=None, data_suffix='.npz'):
        assert os.path.isfile( label_fp ), "File path provided for the labels does not exist. Path iput: {}".format(label_fp)
        self._data_partition = data_partition
        self._data_dir = data_dir
        self._data_suffix = data_suffix

        self._label_fp = label_fp
        self._annonation_direc = annonation_direc

        self.fps = 25 if modality == "video" else 16000
        self.is_var_length = True
        self.label_idx = -3

        self.preprocessing_func = preprocessing_func

        self._data_files = []

        self.load_dataset()


    def load_dataset(self):

        # -- read the labels file
        self._labels = read_txt_lines(self._label_fp)

        # -- add examples to self._data_files
        self._get_files_for_partition()

        # -- from self._data_files to self.list
        self.list = dict()
        self.instance_ids = dict()

        for i, x in enumerate(self._data_files):
            label = self._get_label_from_path( x )
            self.list[i] = [ x, self._labels.index( label ) ]
            self.instance_ids[i] = self._get_instance_id_from_path( x )

        print('Partition {} loaded'.format(self._data_partition))

    def _get_instance_id_from_path(self, x):
        # for now this works for npz/npys, might break for image folders
        instance_id = x.split('/')[-1]
        return os.path.splitext( instance_id )[0]

    def _get_label_from_path(self, x):
        return x.split('/')[self.label_idx]

    def _get_files_for_partition(self):  ##### 여기 확인!!
        # get rgb/mfcc file paths

        dir_fp = self._data_dir
        if not dir_fp:
            return

        # get npy/npz/mp4 files
        search_str_npz = os.path.join(dir_fp, '*', self._data_partition, '*.npz')   # npz : 여러개의 리스트를 한번에 저장하기 위한 포맷
        search_str_npy = os.path.join(dir_fp, '*', self._data_partition, '*.npy')   # npy : 하나의 numpy array를 저장하기 위한 포맷
        search_str_mp4 = os.path.join(dir_fp, '*', self._data_partition, '*.mp4')   
        self._data_files.extend( glob.glob( search_str_npz ) )   # list.extend() : npz파일명을 _data_files에 추가한다.
        self._data_files.extend( glob.glob( search_str_npy ) )   # list.extend() : npy파일명을 _data_files에 추가한다.
        self._data_files.extend( glob.glob( search_str_mp4 ) )   # list.extend() : mp4파일명을 _data_files에 추가한다.

        # If we are not using the full set of labels, remove examples for labels not used
        self._data_files = [ f for f in self._data_files if f.split('/')[self.label_idx] in self._labels ]


    def load_data(self, filename):

        try:
            if filename.endswith('npz'):    # endswith(문자열) : 해당 문자열로 끝나는지 여부를 true/false로 반환
                # return np.load(filename, allow_pickle=True)['data']
                return np.load(filename)['data']
            elif filename.endswith('mp4'):
                return librosa.load(filename, sr=16000)[0][-19456:]   
                # librosa.load() : wav파일을 읽을 때 사용. librosa로 데이터를 읽으면 범위가 -1 ~ 1로 정규화 된다.
                # sr : sampling rate (주파수 분석 및 파형의 시간 간격을 결정)
                # 비디오의 경우 : 1초에 보이는 프레임이 몇 개인가
                # 오디오의 경우 : 프레임이 아닌 샘플이라고 부른다. 단위는 Hz
                # sr이 높은 것이 음질이 좋다.
                # https://wiserloner.tistory.com/1194
                # 16,000 Hz : 표준 전화 협대역인 8,000 Hz보다 높은 광대역 주파수 확장. VoIP
            else:
                return np.load(filename)    
        except IOError:
            print("Error when reading file: {}".format(filename))
            sys.exit()

    def _apply_variable_length_aug(self, filename, raw_data):
        # read info txt file (to see duration of word, to be used to do temporal cropping)
        info_txt = os.path.join(self._annonation_direc, *filename.split('/')[self.label_idx:] )  # swap base folder
        info_txt = os.path.splitext( info_txt )[0] + '.txt'   # swap extension
        info = read_txt_lines(info_txt)  

        utterance_duration = float( info[4].split(' ')[1] )
        half_interval = int(utterance_duration/2.0 * self.fps)  # num frames of utterance / 2
                
        n_frames = raw_data.shape[0]
        mid_idx = ( n_frames -1 ) // 2   # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = random.randint(0, max(0,mid_idx-half_interval-1))    # random.randint(a,b) chooses in [a,b]
        right_idx = random.randint(min( mid_idx+half_interval+1, n_frames ), n_frames)   

        return raw_data[left_idx:right_idx]


    def __getitem__(self, idx):

        raw_data = self.load_data(self.list[idx][0])
        
        # -- perform variable length on training set
        if ( self._data_partition == 'train' ) and self.is_var_length:
            data = self._apply_variable_length_aug(self.list[idx][0], raw_data)
        else:
            data = raw_data
        
        preprocess_data = self.preprocessing_func(data)
        label = self.list[idx][1]
        
        return preprocess_data, label


    def __len__(self):
        return len(self._data_files)


def pad_packed_collate(batch):
    
    batch = np.array(batch, dtype=object)  # list 라서 numpy 로 변경, 내부 요소 리스트 길이가 달라서 dytpe=object 설정하는 코드 추가
    
    if len(batch) == 1:
        data, lengths, labels_np, = zip(*[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
        data = torch.FloatTensor(data)
        lengths = [data.size(1)]

    if len(batch) > 1:
        data_list, lengths, labels_np = zip(*[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])

        data_np = 0  # data_np 변수 초기화하는 코드 추가

        if data_list[0].ndim == 3:
            max_len, h, w = data_list[0].shape  # since it is sorted, the longest video is the first one
            data_np = np.zeros(( len(data_list), max_len, h, w))
        elif data_list[0].ndim == 1:
            max_len = data_list[0].shape[0]
            data_np = np.zeros( (len(data_list), max_len))
        for idx in range( len(data_np)):
            data_np[idx][:data_list[idx].shape[0]] = data_list[idx]
        data = torch.FloatTensor(data_np)

    labels = torch.LongTensor(labels_np)
    
    return data, lengths, labels
