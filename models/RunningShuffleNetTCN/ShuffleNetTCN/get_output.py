import argparse
import json
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from lipreading.model import Lipreading
from preprocessing.transform import warp_img, cut_patch

from torchvision import transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from preprocessing.utils import * 


STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]
START_IDX = 48
STOP_IDX = 68
CROP_WIDTH = CROP_HEIGHT = 96


@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def load_model(config_path: Path, num_classes=500):
    with config_path.open() as fp:
        config = json.load(fp)
    tcn_options = {
        'num_layers': config['tcn_num_layers'],
        'kernel_size': config['tcn_kernel_size'],
        'dropout': config['tcn_dropout'],
        'dwpw': config['tcn_dwpw'],
        'width_mult': config['tcn_width_mult'],
    }
    return Lipreading(
        num_classes=num_classes,
        tcn_options=tcn_options,
        backbone_type=config['backbone_type'],
        relu_type=config['relu_type'],
        width_mult=config['width_mult'],
        extract_feats=False,
    )


def visualize_probs(vocab, probs, col_width=4, col_height=300):
    num_classes = len(probs)
    out = np.zeros((col_height, num_classes * col_width + (num_classes - 1), 3), dtype=np.uint8)

    for i, p in enumerate(probs):
        x = (col_width + 1) * i
        # cv2.rectangle(out, (x, 0), (x + col_width - 1, round(p * col_height)), (255, 255, 255), 1)  # cv2.rectangle(image, start, end, color, thickness)
        
    top = np.argmax(probs)

    prediction = vocab[top].strip()
    confidence = np.round(probs[top], 3)
    print(f'Prediction: {prediction}')
    print(f'Confidence: {confidence}')

    return out, prediction, confidence


# 인자값을 받아서 처리하는 함수
def load_args(default_config=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=Path, default=Path('configs/lrw_resnet18_mstcn.json'))
    parser.add_argument('--model-path', type=Path, default=Path('models/lrw_resnet18_mstcn_adamw_s3.pth.tar'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--queue-length', type=int, default=29)
    parser.add_argument('--video-data', type=None, default='sample/AFTERNOON.mp4')
    parser.add_argument('--label-path', type=Path, default=Path('labels/500WordsSortedList_backup.txt'))
    parser.add_argument('--save-dir', type=Path, default=Path('result/'))
    args = parser.parse_args()
    return args


# 디렉토리 생성
def make_dir(file_path):
    # 파일 없을 경우                 
    if not os.path.exists(os.path.dirname(file_path)):                            
        os.makedirs(os.path.dirname(file_path))  # 디렉토리 생성


def main():
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = load_args()  # args 파싱 및 로드

    mean_face_landmarks = np.load(Path('preprocessing/20words_mean_face.npy'))

    label_path = args.label_path
    with Path(label_path).open() as fp:
        vocab = fp.readlines()
    # assert len(vocab) == 500
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=args.device)
    
    video_pathname = args.video_data

    # -------- 원본 프레임 저장 --------
    cap = cv2.VideoCapture(video_pathname)
    if not cap.isOpened():
        print("could not open : ", video_pathname)
        cap.release()
        exit(0)

    idx = 0
    while True:
        ret, image_np = cap.read()
        if not ret:
            break
        origin = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        origin_save_path = str(args.save_dir) + f'/origin/origin_{idx}.jpg'
        make_dir(origin_save_path)
        cv2.imwrite(origin_save_path, cv2.cvtColor(origin, cv2.COLOR_RGB2BGR))
        idx += 1
    
    # -------- 원본 프레임 GIF 생성 --------
    origin_frames = []
    for idx in range(len(os.listdir(str(args.save_dir) + f'/origin'))):
        origin_frames.append(Image.open(str(args.save_dir) + f'/origin/origin_{idx}.jpg'))

    gif_name = str(args.save_dir) + f'/origin_GIF.gif'
    origin_frames[0].save(f'{gif_name}', format='GIF',
               append_images=origin_frames[1:],
               save_all=True,
               duration=50, loop=0)


    # -------- 영상 처리 -------- 
    video_info = get_video_info(video_pathname, is_print=False)
    output_video = 0
    target_frames= args.queue_length-1
    # 프레임 개수가 다를 경우 -> 전처리 진행
    if target_frames != video_info['length']:
        model = load_model(args.config_path, num_classes=len(vocab))
        model = model.to(args.device)

        queue = deque(maxlen=args.queue_length)

        video = videoToArray(video_pathname, is_gray=False)  # 영상 정보 앞에 영상 프레임 개수를 추가한 numpy
        output_video = frameAdjust(video, target_frames)  # frame sampling (프레임 개수 맞추기)

        def get_yield(output_video):
            for frame in output_video:
                yield frame
        
        print(f'\n ------------ START ------------ \n')
        landmark_idx = 0
        probs_idx = 0
        for frame_idx, frame in enumerate(get_yield(output_video)):
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            all_landmarks = fa.get_landmarks(image_np)
            if all_landmarks:
                landmarks = all_landmarks[0]

            # BEGIN PROCESSING

            trans_frame, trans = warp_img(
                landmarks[STABLE_PNTS_IDS, :], mean_face_landmarks[STABLE_PNTS_IDS, :], image_np, STD_SIZE)
            trans_landmarks = trans(landmarks)
            patch = cut_patch(
                trans_frame, trans_landmarks[START_IDX:STOP_IDX], CROP_HEIGHT // 2, CROP_WIDTH // 2)
            
            # cv2.imshow('patch', cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            patch_save_path = str(args.save_dir) + f'/patch/patch_{landmark_idx}.jpg'
            make_dir(patch_save_path)
            # cv2.imwrite(patch_save_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            cv2.imwrite(patch_save_path, cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY))
            landmark_idx += 1

            patch = Image.fromarray(np.uint8(patch))  # numpy to image
            img_transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),  # gray
                    transforms.ToTensor(),  # image to tensor
                    transforms.Normalize((0.5,),(0.5,)),  # gray image 를 color image 로 load 하기 위함 # 참고: https://github.com/pytorch/vision/issues/288
                    transforms.Lambda(lambda x: x.to(args.device))
                ]
            )
            patch_torch = img_transform(patch)
            queue.append(patch_torch)
            print(f'------------ FRAME {str(frame_idx).zfill(2)} ------------') 
            
            if len(queue)+1 >= args.queue_length:
                confidence = 0
                print(f'\n ------------ PREDICT ------------ \n')
                with torch.no_grad():
                    model_input = torch.stack(list(queue), dim=1).unsqueeze(0)
                    logits = model(model_input, lengths=[args.queue_length])
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    probs = probs[0].detach().cpu().numpy()

                top = np.argmax(probs)
                prediction = vocab[top].strip()
                confidence = np.round(probs[top], 3)
                print(f'Prediction: {prediction}')
                print(f'Confidence: {confidence}')

                # ------------ predict.txt 저장 -----------
                txt_save_path = str(args.save_dir) + f'/predict.txt'
                make_dir(txt_save_path)
                with open(txt_save_path, 'w', encoding='utf8') as f:
                    f.write(f'Prediction: {prediction}, Confidence: {confidence}\n')

                # ------------ video.srt 저장 -----------
                video_name, ext = video_pathname.split('/')[-1].split('.')
                srt_save_path = str(args.save_dir) + f'/{video_name}.srt'
                make_dir(srt_save_path)
                with open(srt_save_path, 'w', encoding='utf8') as f:
                    f.write(f'1\n00:00:00,000 --> 00:00:01,333\n{prediction}\n')
                
            # END PROCESSING

            for x, y in landmarks:
                cv2.circle(image_np, (int(x), int(y)), 2, (0, 0, 255))
        
            # cv2.imshow('camera', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            camera_save_path = str(args.save_dir) + f'/camera/camera_{frame_idx}.jpg'
            make_dir(camera_save_path)
            cv2.imwrite(camera_save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    # 프레임 개수가 같을 경우
    else:
        model = load_model(args.config_path, num_classes=len(vocab))
        model.load_state_dict(torch.load(Path(args.model_path), map_location=args.device)['model_state_dict'])  # [500,768]
        model = model.to(args.device)

        queue = deque(maxlen=args.queue_length)

        cap = cv2.VideoCapture(video_pathname)
        if not cap.isOpened():
            print("could not open : ", video_pathname)
            cap.release()
            exit(0)

        print(f'\n ------------ START ------------ \n')
        frame_idx = 0
        landmark_idx = 0
        probs_idx = 0
        while True:
            ret, image_np = cap.read()
            if not ret:
                break
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            all_landmarks = fa.get_landmarks(image_np)
            if all_landmarks:
                landmarks = all_landmarks[0]

                # BEGIN PROCESSING

                trans_frame, trans = warp_img(
                    landmarks[STABLE_PNTS_IDS, :], mean_face_landmarks[STABLE_PNTS_IDS, :], image_np, STD_SIZE)
                trans_landmarks = trans(landmarks)
                patch = cut_patch(
                    trans_frame, trans_landmarks[START_IDX:STOP_IDX], CROP_HEIGHT // 2, CROP_WIDTH // 2)
                
                # cv2.imshow('patch', cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                patch_save_path = str(args.save_dir) + f'/patch/patch_{landmark_idx}.jpg'
                make_dir(patch_save_path)
                cv2.imwrite(patch_save_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                landmark_idx += 1

                patch = Image.fromarray(np.uint8(patch))  # numpy to image
                img_transform = transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=1),  # gray
                        transforms.ToTensor(),  # image to tensor
                        transforms.Normalize((0.5,),(0.5,)),  # gray image 를 color image 로 load 하기 위함 # 참고: https://github.com/pytorch/vision/issues/288
                        transforms.Lambda(lambda x: x.to(args.device))
                    ]
                )
                patch_torch = img_transform(patch)
                queue.append(patch_torch)
                print(f' ------------ FRAME {str(frame_idx).zfill(2)} ------------ ') 
                
                if len(queue)+1 >= args.queue_length:
                    print(f'\n ------------ PREDICT ------------ \n')
                    with torch.no_grad():
                        model_input = torch.stack(list(queue), dim=1).unsqueeze(0)
                        logits = model(model_input, lengths=[args.queue_length])
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        probs = probs[0].detach().cpu().numpy()

                    top = np.argmax(probs)
                    prediction = vocab[top].strip()
                    confidence = np.round(probs[top], 3)
                    print(f'Prediction: {prediction}')
                    print(f'Confidence: {confidence}')

                    # ------------ predict.txt 저장 -----------
                    txt_save_path = str(args.save_dir) + f'/predict.txt'
                    make_dir(txt_save_path)
                    with open(txt_save_path, 'w', encoding='utf8') as f:
                        f.write(f'Prediction: {prediction}, Confidence: {confidence}\n')

                    # ------------ video.srt 저장 -----------
                    video_name, ext = video_pathname.split('/')[-1].split('.')
                    srt_save_path = str(args.save_dir) + f'/{video_name}.srt'
                    make_dir(srt_save_path)
                    with open(srt_save_path, 'w', encoding='utf8') as f:
                        f.write(f'1\n00:00:00,000 --> 00:00:01,333\n{prediction}\n')
                        
                # END PROCESSING

                for x, y in landmarks:
                    cv2.circle(image_np, (int(x), int(y)), 2, (0, 0, 255))

            # cv2.imshow('camera', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            camera_save_path = str(args.save_dir) + f'/camera/camera_{frame_idx}.jpg'
            make_dir(camera_save_path)
            cv2.imwrite(camera_save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))            
            frame_idx += 1
    
    torch.cuda.empty_cache() # GPU 캐시 데이터 삭제
    cv2.destroyAllWindows()
    print(f'\n ------------ END ------------ \n')


    # ------------ GIF 생성 ------------
    print(f'\n ------------ GIF OUTPUT ------------ ')

    # 원본 프레임 경로
    origin_save_path = str(args.save_dir) + f'/origin/origin_0.jpg'
    make_dir(origin_save_path)

    origin_save_path_list = []
    for idx in range(len(os.listdir(str(args.save_dir) + f'/origin'))):
        origin_save_path_list.append(str(args.save_dir) + f'/origin/origin_{idx}.jpg')

    # 텍스트(prediction) 붙인 프레임 경로
    predict_save_path = str(args.save_dir) + f'/predict/predict_0.jpg'
    make_dir(predict_save_path)

    origin_frames = []
    font_path = '../fonts/NanumGothic.ttf'
    my_font = ImageFont.truetype(font_path, 65)
    for idx, origin_save_path in enumerate(origin_save_path_list):
        # 프레임에 텍스트(prediction) 붙이기
        origin_frame = Image.open(origin_save_path)
        origin_draw = ImageDraw.Draw(origin_frame)
        height, width = origin_frame.size
        origin_draw.text((width//4,height//4), prediction, font=my_font, fill=(255,0,0))
        origin_frame.save(str(args.save_dir) + f'/predict/predict_{idx}.jpg')  # 텍스트(prediction) 붙인 프레임 저장
        
        # 텍스트(prediction) 붙인 프레임 불러오기
        origin_frame = Image.open(str(args.save_dir) + f'/predict/predict_{idx}.jpg')
        origin_frames.append(origin_frame)
    
    # -------- 텍스트(prediction) 붙인 이미지 프레임으로 GIF 생성 --------
    gif_name = str(args.save_dir) + f'/predict_GIF.gif'
    origin_frames[0].save(f'{gif_name}', format='GIF',
               append_images=origin_frames[1:],
               save_all=True,
               duration=50, loop=0)

    print(f' ------------ GIF DONE ------------ \n')


if __name__ == '__main__':
    main()
