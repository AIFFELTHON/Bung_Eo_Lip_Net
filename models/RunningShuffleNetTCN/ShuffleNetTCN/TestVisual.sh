python main.py \
--config-path ./configs/lrw_resnet18_mstcn.json \
--model-path ./train_logs/tcn/2022-06-06T19:09:00/ckpt.best.pth.tar \
--data-dir ./datasets/visual_data/ \
--label-path ./labels/500WordsSortedList.txt \
--save-dir ./result \
--test
