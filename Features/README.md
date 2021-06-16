# Extract Features fro SoccerNet-v2

Precomputed features are available. This code is meant for completeness and reproducibility.

## Create conda environment

``` bash
conda create -y -n SoccerNet-FeatureExtraction python=3.7
conda activate SoccerNet-FeatureExtraction
conda install -y cudnn cudatoolkit=10.1
pip install scikit-video tensorflow==2.3.0 imutils opencv-python==3.4.11.41 SoccerNet moviepy scikit-learn
```

## Extract ResNET features for all 550 games (500 + 50 challenge)

```bash
python Features/ExtractResNET_TF2.py --soccernet_dirpath /path/to/SoccerNet/ --back_end=TF2 --features=ResNET --video LQ --transform crop --verbose --split all
```

## Reduce features for all 550 games (500 games to estimate PCA + 50 challenge games for inference)

```bash
python Features/ReduceFeaturesPCA.py --soccernet_dirpath /path/to/SoccerNet/
```

## Extract ResNET features for a given video

```bash
python Features/VideoFeatureExtractor.py --path_video <PATH_INPUT_VIDEO> --path_features <PATH_OUTPUT_FEATURES> [--start <START_TIME_SECOND> --duration <DURATION_SECOND> --overwrite --PCA "Features/pca_512_TF2.pkl" --PCA_scaler "Features/average_512_TF2.pkl"]
```
