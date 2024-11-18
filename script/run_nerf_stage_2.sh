#!/bin/bash

list="lego"
# list="chair drums ficus hotdog lego materials mic ship"
root_dir="datasets/nerf_synthetic/"
exp_name="3dgs-neus-best-mask-HP10-no-reg"

for i in $list; do
if [ "$i" = "materials" ]; then
    HP=100
else
    HP=10
fi
python train.py --eval \
-s ${root_dir}${i} \
-t bind \
-m output/NeRF_Syn/${i}/${exp_name} \
--HP 10 \
--N_tri 3 \
--iteration 20000 \
--lambda_mask_entropy 0.1 \
--densification_interval 500 \
--save_training_vis
done
