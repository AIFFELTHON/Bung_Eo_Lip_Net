#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" TCN for lipreading"""

import os
import time
import random
import argparse  # 명령행 인자를 파싱해주는 모듈
import numpy as np
from tqdm import tqdm  # 작업진행률 표시하는 라이브러리

import torch  # 파이토치
import torch.nn as nn  # 클래스 # attribute 를 활용해 state 를 저장하고 활용
import torch.nn.functional as F  # 함수 # 인스턴스화시킬 필요없이 사용 가능

from lipreading.utils import get_save_folder
from lipreading.utils import load_json, save2npz
from lipreading.utils import load_model, CheckpointSaver
from lipreading.utils import get_logger, update_logger_batch
from lipreading.utils import showLR, calculateNorm2, AverageMeter
from lipreading.model import Lipreading
from lipreading.mixup import mixup_data, mixup_criterion
from lipreading.optim_utils import get_optimizer, CosineScheduler
from lipreading.dataloaders import get_data_loaders, get_preprocessing_pipelines

from pathlib import Path
import wandb  # 학습 관리 툴 (Loss, Acc 자동 저장)


# 인자값을 받아서 처리하는 함수
def load_args(default_config=None):
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    
    # 입력받을 인자값 목록
    # -- dataset config
    parser.add_argument('--dataset', default='lrw', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=500, help='Number of classes')
    parser.add_argument('--modality', default='video', choices=['video', 'raw_audio'], help='choose the modality')
    # -- directory
    parser.add_argument('--data-dir', default='./datasets/LRW_h96w96_mouth_crop_gray', help='Loaded data directory')
    parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList.txt', help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default=None, help='Loaded data directory')
    # -- model config
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet'], help='Architecture used for backbone')
    parser.add_argument('--relu-type', type=str, default='relu', choices=['relu','prelu'], help='what relu to use' )
    parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier for mobilenets and shufflenets')
    # -- TCN config
    parser.add_argument('--tcn-kernel-size', type=int, nargs="+", help='Kernel to be used for the TCN module')
    parser.add_argument('--tcn-num-layers', type=int, default=4, help='Number of layers on the TCN module')
    parser.add_argument('--tcn-dropout', type=float, default=0.2, help='Dropout value for the TCN module')
    parser.add_argument('--tcn-dwpw', default=False, action='store_true', help='If True, use the depthwise seperable convolution in TCN architecture')
    parser.add_argument('--tcn-width-mult', type=int, default=1, help='TCN width multiplier')
    # -- train
    parser.add_argument('--training-mode', default='tcn', help='tcn')
    parser.add_argument('--batch-size', type=int, default=4, help='Mini-batch size')  # dafault=32 에서 default=8 (OOM 방지) 로 변경
    parser.add_argument('--optimizer',type=str, default='adamw', choices = ['adam','sgd','adamw'])
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--init-epoch', default=0, type=int, help='epoch to start at')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')  # dafault=80 에서 default=10 (테스트 용도) 로 변경
    parser.add_argument('--test', default=False, action='store_true', help='training mode')
    parser.add_argument('--save-dir', type=Path, default=Path('./result/'))
    # -- mixup
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    # -- test
    parser.add_argument('--model-path', type=str, default=None, help='Pretrained model pathname')
    parser.add_argument('--allow-size-mismatch', default=False, action='store_true',
                        help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
    # -- feature extractor
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')
    parser.add_argument('--mouth-patch-path', type=str, default=None, help='Path to the mouth ROIs, assuming the file is saved as numpy.array')
    parser.add_argument('--mouth-embedding-out-path', type=str, default=None, help='Save mouth embeddings to a specificed path')
    # -- json pathname
    parser.add_argument('--config-path', type=str, default=None, help='Model configuration with json format')
    # -- other vars
    parser.add_argument('--interval', default=50, type=int, help='display interval')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')  # dafault=8 에서 default=2 (GCP core 4개의 절반) 로 변경
    # paths
    parser.add_argument('--logging-dir', type=str, default='./train_logs', help = 'path to the directory in which to save the log file')

    # 입력받은 인자값을 args에 저장 (type: namespace)
    args = parser.parse_args()
    return args


args = load_args()  # args 파싱 및 로드

# 실험 재현을 위해서 난수 고정
torch.manual_seed(1)  # 메인 프레임워크인 pytorch 에서 random seed 고정
np.random.seed(1)  # numpy 에서 random seed 고정
random.seed(1)  # python random 라이브러리에서 random seed 고정

# 참고: 실험 재현하려면 torch.backends.cudnn.deterministic = True, torch.backends.cudnn.benchmark = False 이어야 함
torch.backends.cudnn.benchmark = True  # 내장된 cudnn 자동 튜너를 활성화하여, 하드웨어에 맞게 사용할 최상의 알고리즘(텐서 크기나 conv 연산에 맞게)을 찾음


# feature 추출
def extract_feats(model):
    """
    :rtype: FloatTensor
    """
    model.eval()  # evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수
    preprocessing_func = get_preprocessing_pipelines()['test']  # test 전처리
    
    mouth_patch_path = args.mouth_patch_path.replace('.','')
    dir_name = os.path.dirname(os.path.abspath(__file__))
    dir_name = dir_name + mouth_patch_path
    
    data_paths = [os.path.join(pth, f) for pth, dirs, files in os.walk(dir_name) for f in files]
    
    npz_files = np.load(data_paths[0])['data']
    
    data = preprocessing_func(npz_files)  # data: TxHxW
    # data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW
    return data_paths[0], model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])
    # return model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])


# 평가
def evaluate(model, dset_loader, criterion, is_print=False):

    model.eval()  # evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수

    running_loss = 0.
    running_corrects = 0.

    # evaluation/validation 과정에선 보통 model.eval()과 torch.no_grad()를 함께 사용함
    with torch.no_grad():
        inferences = []
        for batch_idx, (input, lengths, labels) in enumerate(tqdm(dset_loader)):
            # 모델 생성
            # input 텐서의 차원을 하나 더 늘리고 gpu 에 할당
            logits = model(input.unsqueeze(1).cuda(), lengths=lengths)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)  # softmax 적용 후 각 원소 중 최대값 가져오기
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()  # 정확도 계산

            loss = criterion(logits, labels.cuda())  # loss 계산
            running_loss += loss.item() * input.size(0)  # loss.item(): loss 가 갖고 있는 scalar 값
        
            # ------------ Prediction, Confidence 출력 ------------ 

            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs = probs[0].detach().cpu().numpy()

            label_path = args.label_path
            with Path(label_path).open() as fp:
                vocab = fp.readlines()

            top = np.argmax(probs)
            prediction = vocab[top].strip()
            confidence = np.round(probs[top], 3)
            inferences.append({
                'prediction': prediction,
                'confidence': confidence
            })

            if is_print:
                print()
                print(f'Prediction: {prediction}')
                print(f'Confidence: {confidence}')
                print()

    # ------------ Prediction, Confidence 텍스트 파일 저장 ------------ 
    txt_save_path = str(args.save_dir) + f'/predict.txt'
    # 파일 없을 경우                 
    if not os.path.exists(os.path.dirname(txt_save_path)):                            
        os.makedirs(os.path.dirname(txt_save_path))  # 디렉토리 생성
    with open(txt_save_path, 'w') as f:
        for inference in inferences:
            prediction = inference['prediction']
            confidence = inference['confidence']
            f.writelines(f'Prediction: {prediction}, Confidence: {confidence}\n')

    print('Test Dataset {} In Total \t CR: {}'.format( len(dset_loader.dataset), running_corrects/len(dset_loader.dataset)))  # 데이터개수, 정확도 출력
    return running_corrects/len(dset_loader.dataset), running_loss/len(dset_loader.dataset), inferences  # 정확도, loss, inferences 반환


# 모델 학습
def train(wandb, model, dset_loader, criterion, epoch, optimizer, logger):
    data_time = AverageMeter()  # 평균, 현재값 저장
    batch_time = AverageMeter()  # 평균, 현재값 저장

    lr = showLR(optimizer)  # LR 변화값

    # 로거 INFO 작성
    logger.info('-' * 10)
    logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))  # epoch 작성
    logger.info('Current learning rate: {}'.format(lr))  # learning rate 작성

    model.train()  # train mode
    running_loss = 0.
    running_corrects = 0.
    running_all = 0.

    end = time.time()  # 현재 시각
    for batch_idx, (input, lengths, labels) in enumerate(dset_loader):
        # measure data loading time
        data_time.update(time.time() - end)  # 평균, 현재값 업데이트

        # --
        # mixup augmentation 계산
        input, labels_a, labels_b, lam = mixup_data(input, labels, args.alpha)
        labels_a, labels_b = labels_a.cuda(), labels_b.cuda()  # tensor 를 gpu 에 할당

        # Pytorch에서는 gradients값들을 추후에 backward를 해줄때 계속 더해주기 때문
        optimizer.zero_grad()  # 항상 backpropagation을 하기전에 gradients를 zero로 만들어주고 시작을 해야 함

        # 모델 생성
        # input 텐서의 차원을 하나 더 늘리고 gpu 에 할당
        logits = model(input.unsqueeze(1).cuda(), lengths=lengths)

        loss_func = mixup_criterion(labels_a, labels_b, lam)  # mixup 적용
        loss = loss_func(criterion, logits)  # loss 계산

        loss.backward()  # gradient 계산
        optimizer.step()  # 저장된 gradient 값을 이용하여 파라미터를 업데이트

        # measure elapsed time # 경과 시간 측정
        batch_time.update(time.time() - end)  # 평균, 현재값 업데이트
        end = time.time()  # 현재 시각
        # -- compute running performance # 컴퓨팅 실행 성능
        _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)  # softmax 적용 후 각 원소 중 최대값 가져오기
        running_loss += loss.item()*input.size(0)  # loss.item(): loss 가 갖고 있는 scalar 값
        running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()  # 정확도 계산
        running_all += input.size(0)


        # ------------------ wandb 로그 입력 ------------------
        wandb.log({'loss': running_loss, 'acc': running_corrects}, step=epoch)


        # -- log intermediate results # 중간 결과 기록
        if batch_idx % args.interval == 0 or (batch_idx == len(dset_loader)-1):
            # 로거 INFO 작성
            update_logger_batch( args, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all, batch_time, data_time )

    return model  # 모델 반환


# model 설정에 대한 json 작성
def get_model_from_json():
    # json 파일이 있는지 확인, 없으면 AssertionError 메시지를 띄움
    assert args.config_path.endswith('.json') and os.path.isfile(args.config_path), \
        "'.json' config path does not exist. Path input: {}".format(args.config_path)  # 원하는 조건의 변수값을 보증하기 위해 사용

    args_loaded = load_json( args.config_path)  # json 읽어오기
    args.backbone_type = args_loaded['backbone_type']  # json 에서 backbone_type 가져오기
    args.width_mult = args_loaded['width_mult']  # json 에서 width_mult 가져오기
    args.relu_type = args_loaded['relu_type']  # json 에서 relu_type 가져오기

    # TCN 옵션 설정
    tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                    'kernel_size': args_loaded['tcn_kernel_size'],
                    'dropout': args_loaded['tcn_dropout'],
                    'dwpw': args_loaded['tcn_dwpw'],
                    'width_mult': args_loaded['tcn_width_mult'],
                  }
    
    # 립리딩 모델 생성
    model = Lipreading( modality=args.modality,
                        num_classes=args.num_classes,
                        tcn_options=tcn_options,
                        backbone_type=args.backbone_type,
                        relu_type=args.relu_type,
                        width_mult=args.width_mult,
                        extract_feats=args.extract_feats).cuda()
    calculateNorm2(model)  # 모델 학습이 잘 진행되는지 확인 - 일반적으로 parameter norm(L2)은 학습이 진행될수록 커져야 함
    return model  # 모델 반환


# main() 함수
def main():

    # wandb 연결
    wandb.init(project="Lipreading_using_TCN_running", entity="hronaie")
    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size
        }
    
    
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"  # GPU 선택 코드 추가

    # -- logging
    save_path = get_save_folder( args)  # 저장 디렉토리
    print("Model and log being saved in: {}".format(save_path))  # 저장 디렉토리 경로 출력
    logger = get_logger(args, save_path)  # 로거 생성 및 설정
    ckpt_saver = CheckpointSaver(save_path)  # 체크포인트 저장 설정

    # -- get model
    model = get_model_from_json()
    # -- get dataset iterators
    dset_loaders = get_data_loaders(args)
    # -- get loss function
    criterion = nn.CrossEntropyLoss()
    # -- get optimizer
    optimizer = get_optimizer(args, optim_policies=model.parameters())
    # -- get learning rate scheduler
    scheduler = CosineScheduler(args.lr, args.epochs)  # 코사인 스케줄러 설정
    
    if args.model_path:
        # tar 파일이 있는지 확인, 없으면 AssertionError 메시지를 띄움
        assert args.model_path.endswith('.tar') and os.path.isfile(args.model_path), \
            "'.tar' model path does not exist. Path input: {}".format(args.model_path)  # 원하는 조건의 변수값을 보증하기 위해 사용
        # resume from checkpoint
        if args.init_epoch > 0:
            model, optimizer, epoch_idx, ckpt_dict = load_model(args.model_path, model, optimizer)  # 모델 불러오기
            args.init_epoch = epoch_idx  # epoch 설정
            ckpt_saver.set_best_from_ckpt(ckpt_dict)  # best 체크포인트 저장
            logger.info('Model and states have been successfully loaded from {}'.format( args.model_path ))  # 로거 INFO 작성
        # init from trained model
        else:
            model = load_model(args.model_path, model, allow_size_mismatch=args.allow_size_mismatch)  # 모델 불러오기
            logger.info('Model has been successfully loaded from {}'.format( args.model_path ))  # 로거 INFO 작성
        # feature extraction
        if args.mouth_patch_path:
                        
            filename, embeddings = extract_feats(model)
            filename = filename.split('/')[-1]
            save_npz_path = os.path.join(args.mouth_embedding_out_path, filename)
            
            # ExtractEmbedding 은 코드 수정이 필요함!
            save2npz(save_npz_path, data = embeddings.cpu().detach().numpy())  # npz 파일 저장
            # save2npz( args.mouth_embedding_out_path, data = extract_feats(model).cpu().detach().numpy())  # npz 파일 저장
            return
        # if test-time, performance on test partition and exit. Otherwise, performance on validation and continue (sanity check for reload)
        if args.test:
            acc_avg_test, loss_avg_test, inferences = evaluate(model, dset_loaders['test'], criterion, is_print=True)  # 모델 평가

            logging_sentence = 'Test-time performance on partition {}: Loss: {:.4f}\tAcc:{:.4f}'.format( 'test', loss_avg_test, acc_avg_test)
            logger.info(logging_sentence)  # 로거 INFO 작성

            return

    # -- fix learning rate after loading the ckeckpoint (latency)
    if args.model_path and args.init_epoch > 0:
        scheduler.adjust_lr(optimizer, args.init_epoch-1)  # learning rate 업데이트


    epoch = args.init_epoch  # epoch 초기화
    while epoch < args.epochs:
        model = train(wandb, model, dset_loaders['train'], criterion, epoch, optimizer, logger)  # 모델 학습
        acc_avg_val, loss_avg_val, inferences = evaluate(model, dset_loaders['val'], criterion)  # 모델 평가
        logger.info('{} Epoch:\t{:2}\tLoss val: {:.4f}\tAcc val:{:.4f}, LR: {}'.format('val', epoch, loss_avg_val, acc_avg_val, showLR(optimizer)))  # 로거 INFO 작성
        # -- save checkpoint # 체크포인트 상태 기록
        save_dict = {
            'epoch_idx': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        ckpt_saver.save(save_dict, acc_avg_val)  # 체크포인트 저장
        scheduler.adjust_lr(optimizer, epoch)  # learning rate 업데이트
        epoch += 1

    # -- evaluate best-performing epoch on test partition # test 데이터로 best 성능의 epoch 평가
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)  # best 체크포인트 경로
    _ = load_model(best_fp, model)  # 모델 불러오기
    acc_avg_test, loss_avg_test, inferences = evaluate(model, dset_loaders['test'], criterion)  # 모델 평가
    logger.info('Test time performance of best epoch: {} (loss: {})'.format(acc_avg_test, loss_avg_test))  # 로거 INFO 작성
    torch.cuda.empty_cache()  # GPU 캐시 데이터 삭제


# 해당 모듈이 임포트된 경우가 아니라 인터프리터에서 직접 실행된 경우에만, if문 이하의 코드를 돌리라는 명령
# => main.py 실행할 경우 제일 먼저 호출되는 부분
if __name__ == '__main__':  # 현재 스크립트 파일이 실행되는 상태 파악
    main()  # main() 함수 호출