#!/bin/bash

case="lego"
root_dir="datasets/nerf_synthetic/"
mesh_seq_base="output/NeRF_Syn"

# mesh_seq_dir="final_our_softbody_35K"
# mesh_seq_dir="final_our_local_mani"
mesh_seq_dir="final_our_deform_stretch"
exp_name="3dgs-neus-best-mask-HP10-no-reg"


python train.py --eval \
--eval_dynamic --dyn_mesh_dir ${mesh_seq_base}/${case}/${mesh_seq_dir} \
-s ${root_dir}${case} \
-t bind \
-m output/NeRF_Syn/${case}/${exp_name} \
-c output/NeRF_Syn/${case}/${exp_name}/chkpnt20000.pth \
--N_tri 3 \
--HP 10 \
--iteration 20000

python make_video.py -d output/NeRF_Syn/${case}/${exp_name}/eval/${mesh_seq_dir}
