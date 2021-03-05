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

def closest(element, mylist):
    min_dist = 9e99
    for l in mylist:
        dist = abs(element-l)
        if dist < min_dist:
            min_dist = dist
            closest = l
    return l, min_dist

def compare_results(results, GT, delta=1000, half=1):

    with open(results) as f:
        data_results = json.load(f)
    with open(GT) as f:
        data_GT = json.load(f)

    data_results = [int(ann["position"])
                    for ann in data_results["annotations"]]
    print("number results", len(data_results))
    data_GT = [int(ann["position"])
               for ann in data_GT["annotations"] if f"{half} - " in ann["gameTime"]]
    print("number GT", len(data_GT))

    prec = 0
    for res in data_results:
        this_closest, dist = closest(res, data_GT)
        if dist < delta:
            prec += 1
    prec = prec/len(data_results)

    reca = 0
    for res in data_GT:
        this_closest, dist = closest(res, data_results)
        if dist < delta:
            reca += 1
    reca = reca/len(data_GT)

    return prec, reca


from tqdm import tqdm
import random
# l = getListGames(["test","train","valid"], task="camera-changes")
l = getListGames("test", task="camera-changes")

for algo_name in ["histogram","intensity"]:
# for algo_name in ["edges"]:
    for game in tqdm(random.sample(l, len(l))):
    # for game in getListGames("v1", task="camera-changes"):
        for half in [1,2]:
            filename = f"{SOCCERNET_PATH}/{game}/{half}.mkv"
            outname = f"{SOCCERNET_PATH}/{game}/{half}_camerachanges"

                

            if algo_name == "histogram":
                list_param = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9]
                # list_param = [1.0]
            elif algo_name == "intensity":
                # list_param = [30]
                list_param = [30, 40, 50, 60, 20, 10]




            for param in random.sample(list_param, len(list_param)):
            # for param in list_param:
                algo = algo_name+str(param)
                if not os.path.exists(f"{outname}_{algo}.npy"):
                    print(f"{outname}_{algo}.npy")

                    start_time = time.time()

                    videodata = skvideo.io.vread(filename)
                    print(videodata.shape)
                    videometadata = skvideo.io.ffprobe(filename)
                    frame_rate = videometadata['video']['@avg_frame_rate']
                    print(frame_rate)
                    (num_frames, height, width, _) = videodata.shape

                    
                    print(f"Using Algorithm {algo}")
                    scene_idx = skvideo.measure.scenedet(
                        videodata, method=algo_name, parameter1=param)
                    scene_det = np.zeros((num_frames,))
                    scene_det[scene_idx] = 1
                    print("--- %s seconds ---" % (time.time() - start_time))
                    # print(len(scene_det), scene_det)
                    np.save(f"{outname}_{algo}.npy", scene_det)
                # np.save(f"{outname}_{algo}.npy", scene_det)



                scene_det = np.load(f"{outname}_{algo}.npy")

                data = {}


                annotations = []
                for i, f in tqdm(enumerate(scene_det)):
                    if f and i > 0:
                        # print(f"time = {i/25:.3f}")
                        annotations.append({
                            "change_type": "",
                            "gameTime": "",
                            "label": "",
                            "position": str(int(i*1000/25)),
                            "replay": ""
                        })
                data["annotations"] = annotations

                
                with open(f"{outname}_{algo}.json", 'w') as outfile:
                    json.dump(data, outfile, indent=4)

