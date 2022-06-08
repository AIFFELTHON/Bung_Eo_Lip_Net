python main.py \
--config-path ./configs/lrw_resnet18_mstcn.json \
--annonation-direc ../hangeul/ \
--data-dir ./datasets/visual_data/
# --backbone-type resnet \  # shufflenet
# --relu-type relu \  # prelu
# --batch-size 4 \
# --optimizer adamw \  # adam, sgd
# --lr 3e-4 \  # 3e-4
# --epochs 100
# --num-classes 15