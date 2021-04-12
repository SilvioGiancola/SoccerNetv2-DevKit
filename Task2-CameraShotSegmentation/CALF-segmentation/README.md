# SoccerNetv2-CameraShot-CALF-segmentation

## Baseline for SoccerNet v2

```bash
python src/main.py \
--SoccerNet_path=/path/to/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--model_name=CALF_seg \
--loss_weight_segmentation=1.0 \
--loss_weight_detection=0.0
```
