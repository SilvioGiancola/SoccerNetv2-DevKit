# Player Localization for Action Spotting

This repository contains the code to reproduce the action spotting result of the paper: "Camera Calibration and Player Localization in SoccerNet-v2 and Investigation of their Representations for Action Spotting" with the CALF method using camera calibration and player localization on the 17 action classes of SoccerNet-v2.


```bibtex
@InProceedings{Cioppa2021Camera,
  author = { Cioppa, Anthony and Deliège, Adrien and Magera, Floriane and Giancola, Silvio and Barnich, Olivier and Ghanem, Bernard and Van Droogenbroeck, Marc},
  title = {Camera Calibration and Player Localization in SoccerNet-v2 and Investigation of their Representations for Action Spotting},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2021}
}
```


```bibtex
@InProceedings{Cioppa2020Context,
  author = {Cioppa, Anthony and Deliège, Adrien and Giancola, Silvio and Ghanem, Bernard and Van Droogenbroeck, Marc and Gade, Rikke and Moeslund, Thomas B.},
  title = {A Context-Aware Loss Function for Action Spotting in Soccer Videos},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```

The task is to spot 17 types of actions (goals, corners, fouls, ...) in untrimmed broadcast soccer videos. To do so, a calibration algorithm is used in combination with a player detector to retrieve the position of each player seen in the broadcast. We investigate different representations of this player localization: a top-view, a feature vector and a graph representation. These representations are then used in the CALF network to produce the spotting predictions.

<p align="center"><img src="img/Abstract.png" width="480"></p>


## Getting Started

The following instructions will help you install the required libraries and the dataset to run the code. The code runs in <code>python 3</code> and was tested in a conda environment. Pytorch is used as deep learning library.


### Create environment

To create and setup the conda environment, simply follow the following steps:

```bash
conda create -n CALF-pytorch python=3.8
conda activate CALF-pytorch
conda install pytorch=1.6 torchvision=0.7 cudatoolkit=10.1 -c pytorch
pip install SoccerNet
pip install opencv-python-headless
pip install opencv-contrib-python-headless

apt update #optional
apt install libgl1-mesa-glx #optional
pip install imutils #optional
pip install moviepy #optional
pip install ffmpy #optional

TORCH=1.6.0
CUDA=cu101
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

pip install efficientnet_pytorch
```

### Run training and testing

The code for training and testing the network is located inside the <code>src</code> folder under the name <code>main.py</code>. The architecture of the network depends on the feature representation used. They are described is in the <code>model.py</code>!

<p align="center"><img src="img/network.png" width="640"></p>

To train the network on the patterned classes using the top-view representation in color composition, simply run the following command from the root folder:

```bash
python src/main.py --SoccerNet_path=/path/to/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--num_features=512 \
--model_name=CALF_calibration \
--batch_size 32 \
--evaluation_frequency 20 \
--chunks_per_epoch 18000 \
--backbone_player=3DConv \
--calibration --calibration_field --calibration_cone \
--backbone_feature=2DConv  \
 --feature_multiplier 2 \
--class_split visual
```

Please find the complete list of arguments in the <code>src/main.py</code> file

The weights of the network will be saved in the <code>models/model_name/</code> folder alongside a log file tracing the training parameters and the evaluation of the performances. 

Note that you might experience a higher variance in the final performances than with the original CALF method.

### Using our pre-trained weights

To reproduce our results, we provide two models trained respectively on the visual and non-visual classes (see paper for details).

You can run the inference on the visual classes using:

```bash
python src/main.py --SoccerNet_path=/path/to/SoccerNet/ --features=ResNET_TF2_PCA512.npy --num_features=512 --model_name=Calib_Best_VIS --batch_size 32 --backbone_player=3DConv --calibration --calibration_field --calibration_cone --backbone_feature=2DConv  --feature_multiplier 2 --with_resnet 34 --with_dense --class_split visual --test_only 
```

and on the non-visual classes using:

```bash
python src/main.py --SoccerNet_path=/path/to/SoccerNet/ --features=ResNET_TF2_PCA512.npy --num_features=512 --model_name=Calib_Best_NVIS --batch_size 32 --backbone_feature=2DConv --feature_multiplier 2 --class_split nonvisual --test_only
```

To compute the final average-mAP you can simply use the following average mean:

<code>(8/17)*a_mAP(VIS) + (9/17)*a_mAP(NVIS)</code>.

With our pre-trained weights, you should get:

<code>(8/17)*41.83% + (9/17)*51.28%</code>=46.8%.

Note that since each model predicts actions on a particular set of classes, you will need to merge the predictions of both models. Also, as the results are saved in the <code>outputs/</code> folder, you will need to save the results of the first model in another folder before running the second command line, not to overwrite the first results. If you wish to have a complete set of predictions in a single file, you can merge the prediction files game per game by simple concatenation of the predictions, as the set of classes is separate in each model.


## Authors

* **Anthony Cioppa**, University of Liège (ULiège).
* **Adrien Deliège**, University of Liège (ULiège).
* **Floriane Magera**,  EVS Broadcast Equipment (EVS).
* **Silvio Giancola**, King Abdullah University of Science and Technology (KAUST).

See the [AUTHORS](AUTHORS) file for details.


## License

Apache v2.0
See the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Anthony Cioppa is funded by the FRIA, Belgium.
* This work is supported by the DeepSport project of the Walloon Region, at the University of Liège (ULiège), Belgium.
* This work is also supported by the King Abdullah University of Science and Technology (KAUST) Office of Sponsored Research (OSR).
