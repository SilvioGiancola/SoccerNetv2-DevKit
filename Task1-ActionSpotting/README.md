# Baselines for Action Spotting

This folder contains the different baselines for the action spotting task on SoccerNet-V2.

### Camera Calibration and Player Localization: [Top View](CALF_Calibration) / [GCN](CALF_Calibration_GCN)

This is the code for the paper [Camera Calibration and Player Localization in SoccerNet-v2 and Investigation of their Representations for Action Spotting](https://arxiv.org/pdf/2104.09333.pdf) (CVSports2021), that leverages field and players localization for action spotting.

<p align="center"><img src="CALF_Calibration/img/Abstract.png" width="480"></p>


### [Temporally Aware Pooling](TemporallyAwarePooling): NetVLAD++

This is the code for the paper [Temporally-Aware Feature Pooling for Action Spotting in Video Broadcasts](https://arxiv.org/pdf/2104.06779.pdf) (CVSports2021), that introduces the baseline NetVLAD++, among other temporally-aware feature pooling modules.

<p align="center"><img src="TemporallyAwarePooling/img/Abstract.png" width="480"></p>


### [CALF](CALF): A Context-Aware Loss Function for Action Spotting in Soccer Videos 

A custom loss function is used to explicitly model the temporal context around action spots. The main idea behind this loss is to penalize the frames far-distant from the action and steadily decrease the penalty for the frames gradually closer to the action. The frames just before the action are not penalized to avoid providing misleading information as its occurrence is uncertain. However, those just after the action are heavily penalized as we know for sure that the action has occurred.

<p align="center"><img src="CALF/img/Abstract.png" width="480"></p>

### [Pool](Pool): NetVLAD and MaxPool

Those baseline are based on the pooling methods for action spotting introduced in the original [SoccerNet](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Giancola_SoccerNet_A_Scalable_CVPR_2018_paper.pdf) dataset.

<p align="center"><img src="Pooling/img/Abstract.png" width="480"></p>

