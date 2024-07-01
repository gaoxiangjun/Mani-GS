<h1 align="center">Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh</h1>

<p align="center">
<a href="" target="_blank">Xiangjun Gao<sup>1</sup></a>, 
<a href="" target="_blank">Xiaoyu Li<sup>2</sup></a>, 
<a href="" target="_blank">Yiyu Zhuang<sup>3</sup></a>, 
<a href="" target="_blank">Qi Zhang<sup>2</sup></a>, 
<a href="" target="_blank">Wenbo Hu<sup>2</sup></a>, 
<a href="" target="_blank">Chaopeng Zhang<sup>2<i class="fa fa-envelope"> </i></sup></a>, 
<br>
<a href="" target="_blank">Yao Yao<sup>3<i class="fa fa-envelope"> </i></sup></a></h5>,
<a href="" target="_blank">Ying Shan<sup>2</sup></a>
<a href="" target="_blank">Long Quan<sup>1</sup></a>
<br>
<br><sup>1</sup><b>HKUST</b>, <sup>2</sup><b>Tencent</b>,  <sup>3</sup><b>Nanjing University</b>
</p>

<!-- ### <p align="center">[Project Page ](https://gaoxiangjun.github.io/mani_gs)  |  [ArXiv](https://arxiv.org/abs/2405.17811)</p> -->

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2405.17811-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2405.17811)
[![website](https://img.shields.io/badge/Project-Website-brightgreen)](https://gaoxiangjun.github.io/mani_gs)
<!-- [![Twitter](https://img.shields.io/badge/Twitter-🔥%2036k%20views-b31b1b.svg?style=social&logo=twitter)](https://twitter.com/_akhaliq/status/1768484390873477480) <br> -->
</h5>

🤗 This is the official implementation for the paper *Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh*. 

🤗 **TL;DR:** 
We introduce a Gaussian-Mesh binding strategy with self-adaption for 3DGS manipulation, which can maintain high-quality rendering,  have high tolerance for mesh accuracy and support various types of 3DGS manipulations.

![Alt text](assets/teaser.gif)
## 📣 News
- [24-6-30] 🔥 Training and inference Code is released.
- [24-5-29] 🔥 Mani-GS is released on [arXiv](https://arxiv.org/abs/2405.17811).

<!-- ## 👀 Todo
- [x] Release the [arXiv] version.
- [ ] Code Refactoring (now is also a little dirty, sorry for that). -->

## 🌟 Overview
- [🛠️ Installation](https://github.com/gaoxiangjun/Mani-GS?tab=readme-ov-file#%EF%B8%8F-installation)
- [📦 Data preparation](https://github.com/gaoxiangjun/Mani-GS?tab=readme-ov-file#-data-preparation)
- [🚀 Training and Evaluation](https://github.com/gaoxiangjun/Mani-GS?tab=readme-ov-file#-training-and-evaluation)
- [💫 Manipulation](https://github.com/gaoxiangjun/Mani-GS?tab=readme-ov-file#-manipulation)
- [👍 Acknowledgement](https://github.com/gaoxiangjun/Mani-GS?tab=readme-ov-file#-acknowledgement)
- [📘 Citation](https://github.com/gaoxiangjun/Mani-GS?tab=readme-ov-file#-citation)


## 🛠️ Installation
#### Clone this repo
```shell
https://github.com/gaoxiangjun/Mani-GS.git
```
#### Install dependencies
```shell
# install environment
conda env create --file environment.yml
conda activate mani-gs

# install pytorch=1.12.1 and others
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install torch_scatter==2.1.1
pip install kornia==0.6.12
pip install imageio[ffmpeg]

# install nvdiffrast=0.3.1
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast

# install knn-cuda
git clone https://github.com/unlimblue/KNN_CUDA.git
cd KNN_CUDA
make && make install

# install relightable 3D Gaussian
pip install ./r3dg-rasterization
```

## 📦 Data preparation
####  NeRF Synthetic Dataset
Download the NeRF synthetic dataset from [LINK](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi?usp=drive_link) provided by [NeRF](https://github.com/bmild/nerf).

#### DTU Dataset
For real-world DTU data, we adopt the [Relightable3DGaussian](https://github.com/NJU-3DV/Relightable3DGaussian) pre-processed DTU data, which can be downloaded [here](https://box.nju.edu.cn/f/d9858b670ab9480fb526/?dl=1).  

For evaluation, we use the [NeuMesh](https://www.dropbox.com/scl/fi/0pkd77wkv9wl0m35ozy1l/DTU.zip?dl=0&e=1&file_subpath=%2FDTU&rlkey=5bf1m5lyp7ynx5tkrylnv9hil&st=7xcfqux5) test split, which can be download from [here](https://www.dropbox.com/scl/fi/0pkd77wkv9wl0m35ozy1l/DTU.zip?dl=0&e=1&file_subpath=%2FDTU&rlkey=5bf1m5lyp7ynx5tkrylnv9hil&st=7xcfqux5), and should be put in *datasets* folder.

We organize the Data Structure like this:

```
Mani-GS
├── datasets
    ├── nerf_synthetic
    |   ├── chair
    |   ├── ...
    ├── data_dtu
    |   ├── DTU_scan24
    |   ├── ...
    ├── data_dtu_valnames
    |   ├── dtu_scan24
```


## 🚀 Training and Evaluation
The training is divided into two stages: (1) extracting the mesh from 3DGS using Screened Poisson reconstruction or NeuS; and (2) binding 3D Gaussian to a given triangular mesh. 

Stage 1 is optional, as we will provide the pre-extracted mesh using different methods. The evaluation will be conducted once the Stage 2 training is complete. Please note that this evaluation is only for static evaluation.

We provide our extracted mesh using different methods, which can be downloaded from this Google drive [link](https://drive.google.com/file/d/1Ox0dhiKMfiLc5Rly01h0viJVCd2rHtcr/view?usp=drive_link). Please unzip it into `./output`

NeRF Synthetic dataset:
```
sh script/run_nerf_stage_1.sh # (optional)
sh script/run_nerf_stage_2.sh # neus mesh as default
```
DTU data:
```
sh script/run_dtu_stage_1.sh # (optional)
sh script/run_dtu_stage_2.sh # poisson mesh as default
```

## 💫 Manipulation
#### Data Structure
We also provide a manipuated mesh demo `lego` in the aforementioned Google Drive link.
The provided mesh files are organizea like this:

```
Mani-GS
├── output
    ├── NeRF_Syn
    |   ├── lego
    |   |   |── mesh_neus_decimate.ply
    |   |   |── mesh_poi_clean.ply
    |   |   |── mesh_mc_30K.ply
    |   |   |── mesh_35K.ply
    |   |   |── final_our_deform_stretch
    |   |   |   |── 00_1.obj
    |   |   |   |── ...
    |   |   |   |── 00_20.obj
    |   |   |── final_our_softbody_35K
    |   |   |   |── 00_1.obj
    |   |   |   |── ...
    |   |   |   |── 00_40.obj
    │   │   │── ...
    
```
NeRF Synthetic dataset:
```
sh script/run_nerf_demo.sh
```
DTU data:
```
sh script/run_dtu_demo.sh
```
## 👍 Acknowledgement
Our code is built on [Relightable 3DGS](https://github.com/NJU-3DV/Relightable3DGaussian), we sincerely thank their efforts.

## 📘 Citation
If you find our work useful in your research, please be so kind to cite:
```
@article{gao2024mani,
  title={Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh},
  author={Gao, Xiangjun and Li, Xiaoyu and Zhuang, Yiyu and Zhang, Qi and Hu, Wenbo and Zhang, Chaopeng and Yao, Yao and Shan, Ying and Quan, Long},
  journal={arXiv preprint arXiv:2405.17811},
  year={2024}
}
```
