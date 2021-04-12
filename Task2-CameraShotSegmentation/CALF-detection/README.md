# SoccerNetv2-CameraShot-CALF-segmentation

## Baseline for SoccerNet v2

```bash
python src/main.py \
--SoccerNet_path=/path/to/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--batch_size=32 \
--version=2 \
--chunk_size=24 \
--num_detections=9 \
--receptive_field=8 \
--loss_weight_segmentation=0.0 \
--loss_weight_detection=1.0 \
--model_name=CALF_det \
--max_epochs=400
```
