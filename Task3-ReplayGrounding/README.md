# SoccerNet-v2 - Replay Grounding

This repository contains the code to reproduce the replay grounding result of the paper: "SoccerNet-v2 : A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos" with two three baslines:

 CALF method adapted to the 17 classes of SoccerNet-v2. The SoccerNet-v2 paper can be found here: [SoccerNet-v2 paper](https://arxiv.org/pdf/2011.13367.pdf), and the CALF paper "A Context-Aware Loss Function for Action Spotting in Soccer Videos" here: [CALF paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cioppa_A_Context-Aware_Loss_Function_for_Action_Spotting_in_Soccer_Videos_CVPR_2020_paper.pdf).


```bibtex
@InProceedings{Deliege2020SoccerNetv2,
  author = { Deliège, Adrien and Cioppa, Anthony and Giancola, Silvio and Seikavandi, Meisam J. and Dueholm, Jacob V. and Nasrollahi, Kamal and Ghanem, Bernard and Moeslund, Thomas B. and Van Droogenbroeck, Marc},
  title = {SoccerNet-v2 : A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos},
  booktitle = {CoRR},
  month = {Nov},
  year = {2020}
}
```



The task consists in retrieving the timestamp of the action shown in a
given replay shot within the whole game.

<p align="center"><img src="Images/qualitative_replay.png" width="480"></p>


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


### Baselines
- SoccerNetv2-ReplayGrounding-CALF [[Link]](SoccerNetv2-ReplayGrounding-CALF)
- SoccerNetv2-ReplayGrounding-CALF_more_negative [[Link]](SoccerNetv2-ReplayGrounding-CALF_more_negative)
- SoccerNetv2-ReplayGrounding-NetVLAD-More-Negative [[Link]](SoccerNetv2-ReplayGrounding-NetVLAD-More-Negative)


## Authors

* **Anthony Cioppa**, University of Liège (ULiège).
* **Adrien Deliège**, University of Liège (ULiège).
* **Silvio Giancola**, King Abdullah University of Science and Technology (KAUST).
* **Meisam J. Seikavandi**,  Aalborg University (AAU).
* **Jacob V. Dueholm**,  Aalborg University (AAU).

See the [AUTHORS](AUTHORS) file for details.


## License

Apache v2.0
See the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Anthony Cioppa is funded by the FRIA, Belgium.
* This work is supported by the DeepSport project of the Walloon Region, at the University of Liège (ULiège), Belgium.
* This work is also supported by the King Abdullah University of Science and Technology (KAUST) Office of Sponsored Research (OSR).
* This work is also supported by the Milestone Research Program at Aalborg University.
