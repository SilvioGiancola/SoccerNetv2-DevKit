## Test on external video

If you wish to test a pre-trained network on external videos, first install the following libraries:

```
apt-get install 
pip install scikit-video tensorflow==2.3.0 imutils opencv-python==3.4.11.41 SoccerNet moviepy scikit-learn ffmpy
```

and then run the code using:
```
python inference/main.py --video_path=<path to your video> --model_name CALF_benchmark
```

This will produce 17 temporal images of predictions (one per class, only predictions with confidence scores above 0.34) and a json file containing all predictions in <code>inference/outputs</code>.
