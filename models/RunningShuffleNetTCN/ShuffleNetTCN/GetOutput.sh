python get_output.py \
--config-path configs/lrw_resnet18_mstcn.json \
--model-path train_logs/tcn/2022-06-06T19:09:00/ckpt.best.pth.tar \
--device cuda \
--queue-length 30 \
--video-data ../hangeul/함께/test/함께_00032.avi \
--label-path labels/500WordsSortedList.txt \
--save-dir ../hangeul/GetOutput_함께
