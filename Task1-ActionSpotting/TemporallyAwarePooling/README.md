# Temporally-Aware Feature Pooling for Action Spotting in Video Broadcasts

This is the code for the paper [Temporally-Aware Feature Pooling for Action Spotting in Video Broadcasts](https://arxiv.org/pdf/2104.06779.pdf) (CVSports2021), that introduces the baseline NetVLAD++, among other temporally-aware feature pooling modules.

Please refer to this baseline by citing the following work:

```bibtex
@InProceedings{Giancola_2021_CVPR_Workshops,
  author = {Giancola, Silvio and Ghanem, Bernard},
  title = {Temporally-Aware Feature Pooling for Action Spotting in Video Broadcasts},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2021}
}
```

## Create Environment

```bash
conda create -y -n SoccerNetv2-AdvancedPooling python=3.8
conda activate SoccerNetv2-AdvancedPooling
conda install -y pytorch=1.6 torchvision=0.7 cudatoolkit=10.1 -c pytorch
pip install SoccerNet matplotlib sklearn
```

## Train NetVLAD++

`python src/main.py --SoccerNet_path=path/to/SoccerNet/ --model_name=NetVLAD++`

Replace `path/to/SoccerNet/` with a local path for the SoccerNet dataset. If you do not have a copy of SoccerNet, this code will automatically download SoccerNet (~50GB).

The training runs in ~1h on a GTX1080Ti (11GB) but does not consume more than 2GB, allowing to train up to 5 models simultaneously.

## Inference

We provide a list of 5 models we trained using 5 different seeds. You can run the inference with:

```bash
for run in {0..4}; do
python src/main.py --SoccerNet_path=path/to/SoccerNet/ --model_name=NetVLAD++_run_${run} --test_only
done
```

## Submission for the SoccerNet-v2 Challenge

For the SoccerNet-v2 challenge, we train on an aggregation of the train+val sets, we validate on the test set and infer on the challenge set.

`python src/main.py --SoccerNet_path=path/to/SoccerNet/ --model_name=NetVLAD++_Challenge --split_train train valid --split_valid test --split_test challenge`

## Reproducibility (5 runs)

Our models are reproducible: 53.3+-0.2% Avg-mAP

```bash
for run in {0..4}; do
python src/main.py --SoccerNet_path=path/to/SoccerNet/ --model_name=NetVLAD++_run_${run} --seed ${run}
done
```

## Ablation studies (5 runs each)

### NetVLAD++ with PCA instead of learnable linear layer

Simply use the PCA-reduced features with `--features=ResNET_TF2_PCA512.npy`.
Without the linear layer, we report 50.7% Avg-mAP, a 2.6% drop with respect to NetVLAD++.

```bash
for run in {0..4}; do
python src/main.py --SoccerNet_path=path/to/SoccerNet/ --model_name=NetVLAD++_PCA512_run_${run} --features=ResNET_TF2_PCA512.npy --seed ${run}
done
```

### NetVLAD++ without the temporal awareness

Simply use the NetVLAD pooling module with the non-reduced features with `--pool=NetVLAD`.
Without the temporal awareness, we report 50.2% Avg-mAP, a 3.1% drop with respect to NetVLAD++.

```bash
for run in {0..4}; do
python src/main.py --SoccerNet_path=path/to/SoccerNet/ --model_name=NetVLAD_run_${run} --pool=NetVLAD --seed ${run}
done
```

### NetVLAD++ without the temporal awareness and with PCA instead of learnable linear layer

Without the temporal awareness nor the linear layer, the model is equivalent to NetVLAD with optimized parameters for the NMS and the temproal window T, leading to 48.4%, 4.9% short of NetVLAD++.

```bash
for run in {0..4}; do
python src/main.py --SoccerNet_path=path/to/SoccerNet/ --model_name=NetVLAD_PCA512_run_${run} --features=ResNET_TF2_PCA512.npy --pool=NetVLAD --seed ${run}
done
```

### More encoders

SoccerNet-v2 provide a list of alternative video frame features:

- `--features=ResNET_TF2.npy`: ResNET features from SoccerNet-v2
- `--features=ResNET_TF2_PCA512.npy`: ResNET features from SoccerNet-v2 reduced at dimension 512 with PCA
- `--features=C3D.npy`: C3D features from SoccerNet
- `--features=C3D_PCA512.npy`: C3D features from SoccerNet reduced at dimension 512 with PCA
- `--features=I3D.npy`: I3D features from SoccerNet
- `--features=I3D_PCA512.npy`: I3D features from SoccerNet reduced at dimension 512 with PCA

### More temporally-aware pooling modules

We developed alternative pooling module

- `--pool=NetVLAD`: NetVLAD pooling module
- `--pool=NetVLAD++`: Temporally aware NetVLAD pooling module
- `--pool=NetRVLAD`: NetRVLAD pooling module
- `--pool=NetRVLAD++`: Temporally aware NetRVLAD pooling module
- `--pool=MAX`: MAX pooling module
- `--pool=MAX++`: Temporally aware MAX pooling module
- `--pool=AVG`: AVG pooling module
- `--pool=AVG++`: Temporally aware AVG pooling module
