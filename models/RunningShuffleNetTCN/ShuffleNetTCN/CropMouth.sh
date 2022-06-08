python preprocessing/crop_mouth_from_video.py \
--video-direc ../hangeul/ \
--video-format .avi \
--landmark-direc ./landmarks/hangeul_landmarks/ \
--filename-path ../hangeul/korean_detected_face.csv \
--save-direc ./datasets/visual_data/ \
--mean-face ./preprocessing/20words_mean_face.npy \
--convert-gray