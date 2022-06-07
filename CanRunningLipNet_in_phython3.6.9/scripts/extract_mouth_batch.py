''' 
extract_mouth_batch.py
    This script will extract mouth crop of every single video inside source directory
    while preserving the overall structure of the source directory content.
Usage:
    python extract_mouth_batch.py [source directory] [pattern] [target directory] [face predictor path]
    pattern: *.avi, *.mpg, etc 
Example:
    python scripts/extract_mouth_batch.py evaluation/samples/GRID/ *.mpg TARGET/ common/predictors/shape_predictor_68_face_landmarks.dat
    Will make directory TARGET and process everything inside evaluation/samples/GRID/ that match pattern *.mpg.
'''

from lipnet.lipreading.videos import Video
import os, fnmatch, sys, errno  
from skimage import io

SOURCE_PATH = sys.argv[1]
SOURCE_EXTS = sys.argv[2]
TARGET_PATH = sys.argv[3]

FACE_PREDICTOR_PATH = sys.argv[4]


# 디렉토리 생성
def mkdir_p(path):
    try:
        os.makedirs(path)  # 디렉토리 생성
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# 디렉토리에서 파일이 있는지 검사
def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename  # 파일명

# 영상 이름으로 폴더 생성 후 frame 별로 입술 crop image 저장
for filepath in find_files(SOURCE_PATH, SOURCE_EXTS):
    print ("Processing: {}".format(filepath))
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH).from_video(filepath)  # 영상 가져오기

    filepath_wo_ext = os.path.splitext(filepath)[0]  # 파일명
    target_dir = os.path.join(TARGET_PATH, filepath_wo_ext)  # 파일명으로 폴더 이름 설정
    mkdir_p(target_dir)  # 폴더 생성

    i = 0
    for frame in video.mouth:
        io.imsave(os.path.join(target_dir, "mouth_{0:03d}.png".format(i)), frame)  # 이미지 저장
        i += 1