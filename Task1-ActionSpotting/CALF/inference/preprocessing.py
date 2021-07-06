from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config.classes import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2

# Function to transform the timestamps to vectors
def timestamps2long(output_spotting, video_size, chunk_size, receptive_field):

    start = 0
    last = False
    receptive_field = receptive_field//2

    timestamps_long = torch.zeros([video_size,output_spotting.size()[-1]-2], dtype = torch.float, device=output_spotting.device)-1


    for batch in np.arange(output_spotting.size()[0]):

        tmp_timestamps = torch.zeros([chunk_size,output_spotting.size()[-1]-2], dtype = torch.float, device=output_spotting.device)-1
        
        for i in np.arange(output_spotting.size()[1]):
            tmp_timestamps[torch.floor(output_spotting[batch,i,1]*(chunk_size-1)).type(torch.int) , torch.argmax(output_spotting[batch,i,2:]).type(torch.int) ] = output_spotting[batch,i,0]

        # ------------------------------------------
        # Store the result of the chunk in the video
        # ------------------------------------------

        # For the first chunk
        if start == 0:
            timestamps_long[0:chunk_size-receptive_field] = tmp_timestamps[0:chunk_size-receptive_field]

        # For the last chunk
        elif last:
            timestamps_long[start+receptive_field:start+chunk_size] = tmp_timestamps[receptive_field:]
            break

        # For every other chunk
        else:
            timestamps_long[start+receptive_field:start+chunk_size-receptive_field] = tmp_timestamps[receptive_field:chunk_size-receptive_field]
        
        # ---------------
        # Loop Management
        # ---------------

        # Update the index
        start += chunk_size - 2 * receptive_field
        # Check if we are at the last index of the game
        if start + chunk_size >= video_size:
            start = video_size - chunk_size 
            last = True
    return timestamps_long

# Function to transform the batches to vectors
def batch2long(output_segmentation, video_size, chunk_size, receptive_field):

    start = 0
    last = False
    receptive_field = receptive_field//2

    segmentation_long = torch.zeros([video_size,output_segmentation.size()[-1]], dtype = torch.float, device=output_segmentation.device)


    for batch in np.arange(output_segmentation.size()[0]):

        tmp_segmentation = 1-output_segmentation[batch]


        # ------------------------------------------
        # Store the result of the chunk in the video
        # ------------------------------------------

        # For the first chunk
        if start == 0:
            segmentation_long[0:chunk_size-receptive_field] = tmp_segmentation[0:chunk_size-receptive_field]

        # For the last chunk
        elif last:
            segmentation_long[start+receptive_field:start+chunk_size] = tmp_segmentation[receptive_field:]
            break

        # For every other chunk
        else:
            segmentation_long[start+receptive_field:start+chunk_size-receptive_field] = tmp_segmentation[receptive_field:chunk_size-receptive_field]
        
        # ---------------
        # Loop Management
        # ---------------

        # Update the index
        start += chunk_size - 2 * receptive_field
        # Check if we are at the last index of the game
        if start + chunk_size >= video_size:
            start = video_size - chunk_size 
            last = True
    return segmentation_long



def visualize(detections_numpy, segmentations_numpy, class_num=0):


    detection_NMS = list()
    for detection in detections_numpy:
        detection[detection<0.34]=-1
        detection[detection >= 0.34] = 1.2
        detection_NMS.append(detection)

    counter = 0
    for detection, segmentation in tqdm(zip(detection_NMS, segmentations_numpy)):
        plt.figure(figsize=(16,12))
        ax_1 = plt.subplot(111)
        ax_1.spines["top"].set_visible(False)
        ax_1.spines["bottom"].set_visible(False)
        ax_1.spines["right"].set_visible(False)
        ax_1.spines["left"].set_visible(False)
        ax_1.get_xaxis().tick_bottom()
        ax_1.get_yaxis().tick_left()
        ax_1.set_ylim(0,1.4)
        x = np.arange(detection.shape[0])/120
        ax_1.plot(x,segmentation[:,class_num], color="tab:orange", alpha=0.5, linewidth=3)
        ax_1.plot(x,detection[:,class_num], '*', color="tab:green", markersize=15)
        text_location = np.where(detection[:,class_num] > 1)
        print(text_location)
        for loc in text_location[0]:
            loc_min = int(np.floor(int(loc/2)/60))
            loc_sec = int(loc/2)%60
            ax_1.text(loc_min,1.1,str(loc_min).zfill(2) + ":" + str(loc_sec).zfill(2), fontsize=16,rotation=-90, rotation_mode='anchor')
        plt.yticks([0,0.5,1], fontsize=20)
        plt.xticks([0,10,20,30,40,50],fontsize=20)
        plt.xlabel("Game Time (in minutes)", fontsize=20)
        plt.ylabel("Segmentation Score", fontsize=20, color="tab:orange", alpha=0.75)
        plt.title(INVERSE_EVENT_DICTIONARY_V2[class_num], fontsize=20)
        plt.savefig("inference/outputs/"+str(class_num)+".png")
        plt.close()

def NMS(detections, delta):
    
    # Array to put the results of the NMS
    detections_tmp = np.copy(detections)
    detections_NMS = np.zeros(detections.shape)-1

    # Loop over all classes
    for i in np.arange(detections.shape[-1]):
        # Stopping condition
        while(np.max(detections_tmp[:,i]) >= 0):

            # Get the max remaining index and value
            max_value = np.max(detections_tmp[:,i])
            max_index = np.argmax(detections_tmp[:,i])

            detections_NMS[max_index,i] = max_value

            detections_tmp[int(np.maximum(-(delta/2)+max_index,0)): int(np.minimum(max_index+int(delta/2), detections.shape[0])) ,i] = -1

    return detections_NMS