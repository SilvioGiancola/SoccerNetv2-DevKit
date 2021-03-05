#!/bin/bash

python src/main.py \
--SoccerNet_path=/path/to/SoccerNet \
--features=ResNET_TF2_PCA512.npy \
--batch_size=32 \
--version=2 \
--chunk_size=24 \
--num_detections=9 \
--receptive_field=8 \
--batch_size=1 \
--scheduler ReduceLRonPlateau \
--patience=25 \
--model_name BasicModel_seg \
--criterion=MSE \
"$@"