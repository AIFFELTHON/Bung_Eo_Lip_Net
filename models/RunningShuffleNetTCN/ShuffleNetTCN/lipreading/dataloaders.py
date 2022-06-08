import torch
import numpy as np
from lipreading.preprocess import *
from lipreading.dataset import MyDataset, pad_packed_collate


def get_preprocessing_pipelines(modality='video'):
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    if modality == 'video':
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        # train : 
        preprocessing['train'] = Compose([                       # 여러 개의 preprocess를 사용할 때 Compose()를 사용한다. preprocess.py에 설정되어 있음
                                    Normalize(0.0,255.0),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std) ])

        preprocessing['val'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])

        preprocessing['test'] = preprocessing['val']   # test와 val이 같다

    elif modality == 'raw_audio':

        preprocessing['train'] = Compose([
                                    AddNoise( noise=np.load('./data/babbleNoise_resample_16K.npy')),   # train에만 노이즈를 추가해 준다.
                                    NormalizeUtterance()])

        preprocessing['val'] = NormalizeUtterance()   # z-score 정규화를 수행
        preprocessing['test'] = NormalizeUtterance()

    return preprocessing


def get_data_loaders(args):
    preprocessing = get_preprocessing_pipelines( args.modality)

    # create dataset object for each partition
    dsets = {partition: MyDataset(
                modality=args.modality,
                data_partition=partition,
                data_dir=args.data_dir,
                label_fp=args.label_path,
                annonation_direc=args.annonation_direc,
                preprocessing_func=preprocessing[partition],
                data_suffix='.npz'
                ) for partition in ['train', 'val', 'test']}
    
    dset_loaders = {x: torch.utils.data.DataLoader(
                        dsets[x],
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=pad_packed_collate,
                        pin_memory=True,
                        num_workers=args.workers,
                        worker_init_fn=np.random.seed(1)) for x in ['train', 'val', 'test']}
    
    return dset_loaders
    