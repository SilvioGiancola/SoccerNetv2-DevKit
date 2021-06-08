# Camera Calibration and Player Localization in SoccerNet-v2 and Investigation of their Representations for Action Spotting

This is the code for the paper [Camera Calibration and Player Localization in SoccerNet-v2 and Investigation of their Representations for Action Spotting](https://arxiv.org/pdf/2104.09333.pdf) (CVSports2021), that leverages field and players localization for action spotting.

Please refer to this baseline by citing the following work:

```bibtex
@InProceedings{Cioppa_2021_CVPR_Workshops,
  author = {Cioppa, Anthony and Deli{\`e}ge, Adrien and Magera, Floriane and Giancola, Silvio and Barnich, Olivier and Ghanem, Bernard and Van Droogenbroeck, Marc},
  title = {Camera Calibration and Player Localization in SoccerNet-v2 and Investigation of their Representations for Action Spotting},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2021}
}
```

## Create Environment

```bash
conda create -n CALF-pytorch python=3.8
conda activate CALF-pytorch
conda install pytorch=1.6 torchvision=0.7 cudatoolkit=10.1 -c pytorch
pip install SoccerNet
```

## Install pytorch geometric

```bash
TORCH=1.6.0
CUDA=cu101
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

## Run locally

```
python src/main.py \
--SoccerNet_path=path/to/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--num_features=512 \
--model_name=calib_GCN \
--batch_size 32 \
--evaluation_frequency 20 \
--chunks_per_epoch 18000 \
--model_name=CALF_resGCN-14_cal_25_scaled_resnet_vis_M2_run_${i}  \
--backbone_feature=2DConv \
--backbone_player=resGCN-14 \
--dist_graph_player=25 \
--calibration \
--feature_multiplier 2 \
--class_split visual
```

## Run on cluster (SLURM - 10 runs)

```
for i in {0..9}; do sbatch -J ASpM2VrGCN ibex.sh --model_name=calib_GCN_run_${i}  --backbone_feature=2DConv --backbone_player=resGCN-14 --dist_graph_player=25 --calibration --feature_multiplier 2 --class_split visual; done
```
