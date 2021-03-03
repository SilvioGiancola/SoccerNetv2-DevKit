# SoccerNet-v2 - Replay Grounding - NetVLAD

This repository contains the code to reproduce the replay grounding result of the paper: "SoccerNet-v2 : A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos" with NetVLAD baslines:

 NetVLAD method is utilized within Siamese framework. Additionally we mine the negative samples for each replay (5 negative for the training ) The SoccerNet-v2 paper can be found here: [SoccerNet-v2 paper](https://arxiv.org/pdf/2011.13367.pdf), and the NetVLAD paper.

```bibtex
@InProceedings{Deliege2020SoccerNetv2,
  author = { Deliège, Adrien and Cioppa, Anthony and Giancola, Silvio and Seikavandi, Meisam J. and Dueholm, Jacob V. and Nasrollahi, Kamal and Ghanem, Bernard and Moeslund, Thomas B. and Van Droogenbroeck, Marc},
  title = {SoccerNet-v2 : A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos},
  booktitle = {CoRR},
  month = {Nov},
  year = {2020}
}
```



```bibtex
@InProceedings{Giancola_2018_CVPR_Workshops,
  author = {Giancola, Silvio and Amine, Mohieddine and Dghaily, Tarek and Ghanem, Bernard},
  title = {SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2018}
}
```

The task consists in retrieving the timestamp of the action shown in a
given replay shot within the whole game. 

<p align="center"><img src="Images/qualitative_replay.png" width="480"></p>

<!-- For more information about the CALF method, check out our presentation video. To see more of our work, subscribe to our YouTube channel [__Acad AI Research__](https://www.youtube.com/channel/UCYkYA7OwnM07Cx78iZ6RHig?sub_confirmation=1)

<a href="https://www.youtube.com/watch?v=51cyRDcmO00">
<p align="center"><img src="img/Miniature-context-YouTube.png" width="720"></p>
</a>
 -->
## Getting Started

The following instructions will help you install the required libraries and the dataset to run the code. The code runs in <code>python 3</code> and was tested in a conda environment. Pytorch is used as deep learning library. 


### Create environment

To create and setup the conda environment, simply follow the following steps:

```bash
conda create -n CALF-pytorch python=3.8
conda activate CALF-pytorch
conda install pytorch=1.6 torchvision=0.7 cudatoolkit=10.1 -c pytorch
pip install SoccerNet
```



### Run training and testing

The code for training and testing the network is located inside the <code>src</code> folder under the name <code>main.py</code>. 

To train the network, simply run the following command from the root folder:

```bash
python src/main.py --SoccerNet_path=/path/to/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--num_features=512 \
--model_name=CALF_v2 \
--batch_size 32 \
--evaluation_frequency 20 \
--chunk_size 120 \
--receptive_field 40\
--loop 5 \
--unsimilar_action 0.3 \
--hard_negative_weight 0.4 \
--pooling "NetVLAD"
```
You can save the results in json files for each game by adding this arguments:

```bash 

 --detection_path '/outputs/' \
 --save_results True
```
The weights of the network will be saved in the models/model_name/ folder alongside a log file tracing the training parameters and the evolution of the performances. The predictions for the challenge submission on [EvalAI](https://eval.ai/web/challenges/challenge-page/761/overview) (testset split) will be stored in the outputs folder. To submit your results, simply zip the league folders inside a single zip file.

Note that if you did not download the SoccerNet features beforehand, the code will start by downloading them, which might take a bit of time.


To evaluate the results run the model from the main.py file. This option will re-load the weights of your model and predict the results from the features, and save them in json format in the outputs folder:

```bash
python src/main.py --SoccerNet_path=/path/to/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--num_features=512 \
--model_name=NetVLAD_more_negative \
--test_only \
 --detection_path '/outputs/' \
 --save_results True
```


If you wish to use our pre-trained weights, they are located in models/NetVLAD_more_negative_trained/. You can reproduce our results by running:

```bash
python src/main.py --SoccerNet_path=/path/to/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--num_features=512 \
--model_name=NetVLAD_more_negative_trained \
--test_only \
 --detection_path '/outputs/' \
 --save_results True
```

<!-- For producing the results of the challenge, simply use the --challenge parameter. Of course, the performance won't be computed localy since you don't have access to the labels. Upload the predictions saved in the <code>outputs</code> folder on [EvalAI](https://eval.ai/web/challenges/challenge-page/761/overview) to get your challenge performance.

```bash
python src/main.py --SoccerNet_path=/path/to/SoccerNet/ \
--features=ResNET_TF2_PCA512.npy \
--num_features=512 \
--challenge
--model_name=CALF_v2 \
--test_only
``` -->

