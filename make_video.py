# import imageio.v2 as iio
# pip install imageio[ffmpeg]
import imageio
import os
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser(description="Training script parameters") 

parser.add_argument("-d", "--imgs_dir", type=str, default=None)
args = parser.parse_args()

# # load_dir and video_save_path
load_dir = args.imgs_dir
# WIHTE_BG = True
WIHTE_BG = False

# all images
imgs = os.listdir(load_dir)
imgs = [x for x in imgs if x.endswith(".png")]
# select a view
view_index = "r_30"
video_save_path = f"{load_dir}/demo_{view_index}.mp4" 
imgs = [x for x in imgs if "view_index" in x]
imgs.sort()

# For not zero padding image name
imgs = sorted(imgs, key=lambda x: int(x.split('_')[1].split('.')[0]))
imgs = [imageio.v2.imread(os.path.join(load_dir, f)) for f in imgs]

# with white background
if WIHTE_BG:
    for img in imgs:
        # print(img.max())
        # img[np.where(img[:, :, 3] < 128)] = [255, 255, 255, 255]
        img[np.where(img[:, :, 3] < 200)] = [255, 255, 255, 255]
        # img[:, :, 0:3] = img[:, :, 0:3] + (1 - img[:, :, 3:]) * 255
        # img[:, :, 3:] = 255

imageio.mimsave(video_save_path, imgs, fps=10, quality=9)