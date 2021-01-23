# SoccerNetv2-ActionSpotting-Pooling


```bash
conda create -n SoccerNetv2-Pooling python=3.8
conda activate SoccerNetv2-Pooling
conda install pytorch=1.6 torchvision=0.7 cudatoolkit=10.1 -c pytorch
pip install SoccerNet matplotlib sklearn scikit-video scenedetect opencv-python==4.4.0.46
```

## MaxPooling

```bash
python src/main.py \
--SoccerNet_path=/path/to/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--num_features=512 \
--model_name=MAXPOOL_v2 \
--version 2 \
--batch_size 256 \
--chunk_size 20 \
--patience 10 \
--evaluation_frequency=10 \
--pool=MAX
```

## NetVLAD

```bash
python src/main.py \
--SoccerNet_path=/path/to/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--num_features=512 \
--model_name=NETVLAD_v2 \
--version 2 \
--batch_size 256 \
--chunk_size 20 \
--patience 10 \
--evaluation_frequency=10 \
--pool=NetVLAD
```
