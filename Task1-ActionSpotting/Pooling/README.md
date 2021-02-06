# SoccerNetv2-ActionSpotting-Pooling

Those baseline are based on the pooling methods for action spotting introduced on [SoccerNet](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Giancola_SoccerNet_A_Scalable_CVPR_2018_paper.pdf).

This code is an efficient implementation in Pytorch 1.6, adapted from the original [TensorFlow (v1)](https://github.com/SilvioGiancola/SoccerNet-code).

Please refer to those baseline by citing the following work:

```bibtex
@InProceedings{Giancola_2018_CVPR_Workshops,
  author = {Giancola, Silvio and Amine, Mohieddine and Dghaily, Tarek and Ghanem, Bernard},
  title = {SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2018}
}
```

## Setup the environment

```bash
conda create -n SoccerNetv2-Pooling python=3.8
conda activate SoccerNetv2-Pooling
conda install pytorch=1.6 torchvision=0.7 cudatoolkit=10.1 -c pytorch
pip install SoccerNet matplotlib sklearn scikit-video scenedetect opencv-python==4.4.0.46
```

## Main code

```bash
python src/main.py \
--SoccerNet_path=/media/giancos/Football/SoccerNet/ \  # Path of the SoccerNet main directory where the features are stored
--features=ResNET_TF2_PCA512.npy \      # Feature to use
--num_features=512 \                    # dimension of the features
--model_name=NetVLAD_v2 \               # name of the model and destination for the resutls
--version 2 \                           # Version of SoccerNet
--batch_size 256 \                      # Batch size for training
--chunk_size 20 \                       # Size of the clips/chunks of frames in second
--patience 10 \                         # Patience for the reduction of the LR
--evaluation_frequency=10 \             # Frequency of the evaluation (for classification of clips)
--pool=NetVLAD \                        # Pooling method [NetVLAD/MAX]
--NMS_threshold 0.5 \                   # Theshold for the confidence in NMS
--split_train train \                   # split for training (list)
--split_valid valid \                   # split for validation (list)
--split_test test challenge             # split for testing (list)
```

### Example: MaxPooling on SoccerNet v1 (3 classes - Average-mAP=29.52%)

```bash
python src/main.py \
--SoccerNet_path=/media/giancos/Football/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--num_features=512 \
--model_name=MAXPOOL_v1 \
--version 1 \
--batch_size 256 \
--chunk_size 20 \
--patience 10 \
--evaluation_frequency=10 \
--pool=MAX \
--NMS_threshold 0.0 \
--split_train train \
--split_valid valid \
--split_test test
```

### Example: NetVLAD on SoccerNet v1 (3 classes - Average-mAP=44.67%)

```bash
python src/main.py \
--SoccerNet_path=/media/giancos/Football/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--num_features=512 \
--model_name=NETVLAD_v1 \
--version 1 \
--batch_size 256 \
--chunk_size 20 \
--patience 10 \
--evaluation_frequency=10 \
--pool=NetVLAD \
--NMS_threshold 0.5 \
--split_train train \
--split_valid valid \
--split_test test
```

### Example: MaxPooling on SoccerNet v2 (17 classes - Average-mAP=18.5%)

```bash
python src/main.py \
--SoccerNet_path=/media/giancos/Football/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--num_features=512 \
--model_name=MAXPOOL_v2 \
--version 2 \
--batch_size 256 \
--chunk_size 20 \
--patience 10 \
--evaluation_frequency=10 \
--pool=MAX \
--NMS_threshold 0.0 \
--split_train train \
--split_valid valid \
--split_test test challenge
```

### Example: NetVLAD on SoccerNet v2 (17 classes - Average-mAP=31.37%))

```bash
python src/main.py \
--SoccerNet_path=/media/giancos/Football/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--num_features=512 \
--model_name=NETVLAD_v2 \
--version 2 \
--batch_size 256 \
--chunk_size 20 \
--patience 10 \
--evaluation_frequency=10 \
--pool=NetVLAD \
--NMS_threshold 0.5 \
--split_train train \
--split_valid valid \
--split_test test challenge
```

### Note on performances

The implementation of NetVLAD from SoccerNet-v1 considered a confidence threshold of `0.5` for the NMS.
Setting that value to `0.0` (similar to CALF), lead to an Average-mAP of 41.68%, comparable with the performances  [CALF](Task1-ActionSpotting/CALF/).
