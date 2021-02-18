# Annotation and Visualization Tools

This folder contains the tools used to annotate the actions, the camera shots and replays of SoccerNet-v2.
These tools can also be used to visualize the SoccerNet-v2 annotations on the videos.

## Getting Started

The following instructions will help you install the required libraries and the dataset to run the tools. The code runs in <code>python 3</code> and was tested in a conda environment.

### Create environment

To create and setup the conda environment, simply follow these steps:

```bash
conda create -n annotation python=3.8
conda activate annotation
pip install --upgrade pip
pip install pyqt5
```

### Install the SoccerNet-v2 videos and labels

You will need the LQ videos, the action labels and the camera labels. The following steps help you install them. Note that in order to download the videos, you will need to sign a NDA to get the password on the [SoccerNet website](https://soccer-net.org/).

```bash
pip install SoccerNet
python
>> from SoccerNet.Downloader import SoccerNetDownloader
>> mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/path/to/soccernet")

>> mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"]) # download labels SN v2
>> mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train","valid","test"]) # download labels for camera shot

>> mySoccerNetDownloader.password = input("Password for videos? (contact the author):\n")
>> mySoccerNetDownloader.downloadGames(files=["1.mkv", "2.mkv"], split=["train","valid","test"]) # download LQ Video
```

## Run the tools

Once the environment is ready, you can simply run the annotation tool for action spotting with the following commands:

```bash
cd actions/src/
python main.py
```

or the annotation tool for the camera shots and the replays:

```bash
cd replays/src/
python main.py
```

### Start the visualization

To start the visualization, click on the open video button at the bottom left of the annotator and select a low-quality video from the SoccerNet dataset (named 1.mkv or 2.mkv). The labels-v2.json (for actions) or the Labels-cameras.json (for the camera shots and replays) should be placed in the same folder as the video (done automatically when downloading them from the SoccerNet pip package).

Once the video is opened, you can easily navigate through it either using the time bar at the bottom or by double-clicking on the annotations in the right column. You can also navigate in the video frame by frame using the left and right arrow keys, or speed up the video using A (x1) Z (x2) E (x4) and pause or re-start the video with the SPACEBAR.

### Annotate your own actions or camera shots

To annotate a new action or camera shot, go to the exact frame you want to annotate, press ENTER (not the one on the numpad) and navigate through the menu.

To delete an annotation, double-click on it in the right column and press DELETE. 

Be aware that these actions will overwrite the original annotations of SoccerNet and might thus change the performances of methods. To make sure you have the correct annotations when evaluating your method, either work on a copy of the dataset or re-download the annotations directly from the pip package in a new folder.


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



