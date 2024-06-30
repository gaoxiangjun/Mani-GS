#!/bin/bash

root_dir="datasets/data_dtu/DTU_scan"
list="118"

mesh_seq_base="output/DTU"
mesh_seq_dir="edited-mesh"
exp_name="3dgs-HP10"


for i in $list
do

python train.py --eval \
--eval_dynamic --dyn_mesh_dir ${mesh_seq_base}/${i}/${mesh_seq_dir} \
-s ${root_dir}${i} \
-t bind \
-m output/DTU/${i}/3dgs-HP10 \
-c output/DTU/${i}/3dgs-HP10/chkpnt10000.pth \
--iteration 10000 \

python make_video.py -d output/DTU/${i}/${exp_name}/eval/${mesh_seq_dir}

done
