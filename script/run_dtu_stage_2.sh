#!/bin/bash

root_dir="datasets/data_dtu/DTU_scan"
list="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
# list="118"

for i in $list
do
python train.py --eval \
-s ${root_dir}${i} \
-t bind \
-m output/DTU/${i}/3dgs-HP10-NeuMesh-val-list \
--lambda_normal_render_depth 0.0 \
--lambda_mask_entropy 0.1 \
--lambda_depth 0 \
--iteration 20000 \
--lambda_normal_mvs_depth 0.0 \
--densification_interval 500 \
--save_training_vis
done
