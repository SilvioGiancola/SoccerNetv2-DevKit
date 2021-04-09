# Baselines for Camera Shot Segmentation and Camera Change Detection

This folder contains the different baselines for the action spotting task on SoccerNet-V2.

### [BasicModel-segmentation](BasicModel-segmentation): Based on CALF

This network is based on the segmentation module of the network introduced in "A Context-Aware Loss Function for Action Spotting in Soccer Videos", where only the segmentation module is kept and the spotting module discarded. The training loss is also adapted for the task of camera segmentation.

### [PySceneDetect-detection](PySceneDetect-detection): Based on PySceneDetect

This baseline is based on the [PySceneDetect](https://pypi.org/project/scenedetect/) library, looking for changes or cuts in videos.

### [ScikitVideo-detection](ScikitVideo-detection): Based on scikit-video

This baseline is based on the [scikit-video](https://pypi.org/project/scikit-video/) library.

