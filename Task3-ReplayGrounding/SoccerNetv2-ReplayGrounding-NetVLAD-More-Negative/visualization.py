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
        detections_numpy.append(detection.numpy())
        segmentations_numpy.append(segmentation.numpy())
        replays_numpy.append(replay.numpy())

    counter = 0
    for target, detection, segmentation, replay in tqdm(zip(targets_numpy,detections_numpy, segmentations_numpy, replays_numpy)):
        print(target.shape)
        print(detection.shape)
        x = np.arange(target.shape[0])
        plt.ylim(-0.1,1.1)
        plt.plot(x,target[:,0], color="tab:blue")
        plt.plot(x,replay[:,0], color="tab:red")
        plt.plot(x,segmentation[:,0], color="tab:orange", alpha=0.75)
        plt.plot(x,detection[:,0], '*', color="tab:green")
        plt.savefig("fig/"+str(counter)+".png")
        plt.close()
        counter += 1