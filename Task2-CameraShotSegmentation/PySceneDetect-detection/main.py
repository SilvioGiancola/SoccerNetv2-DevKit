from tqdm import tqdm
import time
# import matplotlib.pyplot as plt
import numpy as np

from SoccerNet.utils import getListGames


import skvideo.measure
import skvideo.io
import skvideo.datasets
import json 
import os 

try:
    xrange
except NameError:
    xrange = range

SOCCERNET_PATH = "/path/to/SoccerNet"


# Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector, ThresholdDetector

def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    # Base timestamp at frame 0 (required to obtain the scene list).
    base_timecode = video_manager.get_base_timecode()
    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()
    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list(base_timecode)


from tqdm import tqdm
import random
# l = getListGames(["test","train","valid"], task="camera-changes")
l = getListGames("test", task="camera-changes")

# video_path = []
for algo_name in ["ContentDetector"]:
    for game in tqdm(random.sample(l, len(l))):
    # for game in getListGames("v1", task="camera-changes"):
        for half in [1,2]:
            filename = f"{SOCCERNET_PATH}/{game}/{half}.mkv"
            outname = f"{SOCCERNET_PATH}/{game}/{half}_camerachanges"
            # video_path.append(filename)
# ThresholdDetector
# ThresholdDetector
            if algo_name == "ContentDetector":
                list_param = [10,20,30,40,50,60]

            for threshold in random.sample(list_param, len(list_param)):
                algo = f"{algo_name}{threshold}"
                if os.path.exists(f"{outname}_{algo}.json"):
                    continue
                video_manager = VideoManager([filename])
                scene_manager = SceneManager()
            # for threshold in [10,20,30,40,50,60]:

                if algo_name == "ContentDetector":
                    scene_manager.add_detector(
                        ContentDetector(threshold=threshold))
                
                # list_param = [64, 128, 256, 32, 16, 8, 4]

                # Base timestamp at frame 0 (required to obtain the scene list).
                base_timecode = video_manager.get_base_timecode()

                # Improve processing speed by downscaling before processing.
                video_manager.set_downscale_factor()

                # Start the video manager and perform the scene detection.
                video_manager.start()
                scene_manager.detect_scenes(frame_source=video_manager)

                # Each returned scene is a tuple of the (start, end) timecode.
                results = scene_manager.get_scene_list(base_timecode)
                # print(" ")
                # print(" ")
                # print(results)
                print(f"{algo} found {len(results)} shots")

                data = {}


                annotations = []
                for i, res in enumerate(results):
                    f = res[1]
                    # print("f", f.get_seconds())
                    if f and i > 0:
                        # print(f"time = {i/25:.3f}")
                        annotations.append({
                            "change_type": "",
                            "gameTime": "",
                            "label": "",
                            "position": str(int(f.get_seconds()*1000)),
                            "replay": ""
                        })
                data["annotations"] = annotations
                # print(data)

            
                with open(f"{outname}_{algo}.json", 'w') as outfile:
                    json.dump(data, outfile, indent=4)

