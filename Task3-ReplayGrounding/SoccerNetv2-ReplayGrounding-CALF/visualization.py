from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def visualize(targets, detections, segmentations, replays):

    targets_numpy = list()
    detections_numpy = list()
    segmentations_numpy = list()
    replays_numpy = list()
    
    for target, detection, segmentation, replay in zip(targets,detections, segmentations, replays):
        targets_numpy.append(target.numpy())
        arr = detection.numpy()
        arr[arr < 0.5] = -1
        arr[arr >= 0.5] = 1.2
        detections_numpy.append(arr)
        segmentations_numpy.append(segmentation.numpy())
        replays_numpy.append(replay.numpy())

    counter = 0
    for target, detection, segmentation, replay in tqdm(zip(targets_numpy,detections_numpy, segmentations_numpy, replays_numpy)):
        plt.figure(figsize=(16,12))
        ax_1 = plt.subplot(111)
        ax_1.spines["top"].set_visible(False)
        ax_1.spines["bottom"].set_visible(False)
        ax_1.spines["right"].set_visible(False)
        ax_1.spines["left"].set_visible(False)
        ax_1.get_xaxis().tick_bottom()
        ax_1.get_yaxis().tick_left()
        ax_1.set_ylim(0,1.4)
        plt.yticks([0,0.5,1], fontsize=20)
        plt.xticks([0,10,20,30,40,50], fontsize=20)
        plt.xlabel("Game Time (in minutes)", fontsize=20)
        plt.ylabel("Segmentation Score", fontsize=20, color="tab:orange", alpha=0.75)
        print(target.shape)
        print(detection.shape)
        x = np.arange(target.shape[0])/120
        plt.plot(x,target[:,0], color="tab:blue", linewidth=4)
        plt.plot(x,replay[:,0], color="tab:red")
        plt.plot(x,segmentation[:,0], color="tab:orange", alpha=0.5, linewidth=3)
        plt.plot(x,detection[:,0], '*', color="tab:green", markersize=15)
        plt.savefig("fig/"+str(counter)+".png")
        plt.close()
        counter += 1
