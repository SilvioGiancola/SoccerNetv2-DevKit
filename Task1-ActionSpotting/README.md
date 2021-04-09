# Baselines for Action Spotting

This folder contains the different baselines for the action spotting task on SoccerNet-V2.

### [CALF](CALF): A Context-Aware Loss Function for Action Spotting in Soccer Videos 

a custom loss function is used to explicitly model the temporal context around action spots. The main idea behind this loss is to penalize the frames far-distant from the action and steadily decrease the penalty for the frames gradually closer to the action. The frames just before the action are not penalized to avoid providing misleading information as its occurrence is uncertain. However, those just after the action are heavily penalized as we know for sure that the action has occurred.

<p align="center"><img src="CALF/img/Abstract.png" width="480"></p>

### [Pool](Pool): NetVLAD and MaxPool

Those baseline are based on the pooling methods for action spotting introduced in the original [SoccerNet](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Giancola_SoccerNet_A_Scalable_CVPR_2018_paper.pdf) dataset.
