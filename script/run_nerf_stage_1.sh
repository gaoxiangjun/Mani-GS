#!/bin/bash

list="lego"
# list="chair drums ficus hotdog lego materials mic ship"
root_dir="datasets/nerf_synthetic/"

for i in $list; do
python train.py \
-s ${root_dir}${i} \
-t render \
-m output/NeRF_Syn/${i}/3dgs \
--iteration 30000 \
--lambda_normal_render_depth 0.01 \
--lambda_mask_entropy 0.1 \
--densification_interval 500 \
--save_training_vis
done
