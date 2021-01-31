# SoccerNetv2-DevKit

Development Kit for the SoccerNet Challenge. This kit is meant as a help to get started working with data see https://soccer-net.org/

SoccerNet-v2 is an extension of SoccerNet-v1 with new and challenging tasks including
action spotting, camera shot segmentation with boundary detection, and a novel replay grounding task.
The dataset consists of 500 soccer games including:
 - Videos in both low and high resolution.
 - Pre-computed features.
 - Annotations of actions (Labels-v2.json).
 - Annotations of camera replays linked to actions (Labels-cameras.json).
 - Annotations of camera changes (Labels-cameras.json) (200 games).


## How to download SoccerNet-v2 [[Link]](Download)


## How to extract video features [[Link]](Features)


## Baseline Implementations

- Action Spotting [[Link]](Task1-ActionSpotting)
- Camera Shot Segmentation [[Link]](Task2-CameraShotSegmentation)
- Replay Grounding [[Link]](Task3-ReplayGrounding)


## Evaluation [[Link]](Evaluation)


## Citation

For further information check out the paper and supplementary material:
https://arxiv.org/abs/2011.13367

Please cite our work if you use our dataset:
```bibtex
@misc{deliège2020soccernetv2,
      title={SoccerNet-v2 : A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos}, 
      author={Adrien Deliège and Anthony Cioppa and Silvio Giancola and Meisam J. Seikavandi and Jacob V. Dueholm and Kamal Nasrollahi and Bernard Ghanem and Thomas B. Moeslund and Marc Van Droogenbroeck},
      year={2020},
      eprint={2011.13367},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
