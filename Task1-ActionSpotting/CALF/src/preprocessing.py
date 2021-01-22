
import numpy as np
import torch

def rulesToCombineShifts(shift_from_last_event, shift_until_next_event, params):
    
    s1  = shift_from_last_event
    s2  = shift_until_next_event
    K = params
    
    if s1 < K[2]:
        value = s1
    elif s1 < K[3]:
        if s2 <= K[0]:
            value = s1
        else:
            if (s1-K[2])/(K[3]-K[2]) < (K[1]-s2)/(K[1]-K[0]):
                value = s1
            else:
                value = s2
    else:
        value = s2
        
    return value

def oneHotToShifts(onehot, params):

    
    nb_frames = onehot.shape[0]
    nb_actions = onehot.shape[1]
    
    Shifts = np.empty(onehot.shape)
    
    for i in range(nb_actions):
        
        x = onehot[:,i]
        K = params[:,i]
        shifts = np.empty(nb_frames)
        
        loc_events = np.where(x == 1)[0]
        nb_events = len(loc_events)
        
        if nb_events == 0:
            shifts = np.full(nb_frames, K[0])
        elif nb_events == 1:
            shifts = np.arange(nb_frames) - loc_events
        else:
            loc_events = np.concatenate(([-K[3]],loc_events,[nb_frames-K[0]]))
            for j in range(nb_frames):
                shift_from_last_event = j - loc_events[np.where(j >= loc_events)[0][-1]]
                shift_until_next_event = j - loc_events[np.where(j < loc_events)[0][0]]
                shifts[j] = rulesToCombineShifts(shift_from_last_event, shift_until_next_event, K)
        
        Shifts[:,i] = shifts
    
    return Shifts

import random


def getNegativeIndexes(labels, params, chunk_size):

    zero_one_labels = np.zeros(labels.shape)
    for i in np.arange(labels.shape[1]):
        zero_one_labels[:,i] = 1-np.logical_or(np.where(labels[:,i] >= params[3,i], 1,0),np.where(labels[:,i] <= params[0,i], 1,0))
    zero_one = np.where(np.sum(zero_one_labels, axis=1)>0, 0, 1)

    zero_one_pad = np.append(np.append([1-zero_one[0],], zero_one, axis=0), [1-zero_one[-1]], axis=0)
    zero_one_pad_shift = np.append(zero_one_pad[1:], zero_one_pad[-1])

    zero_one_sub = zero_one_pad - zero_one_pad_shift

    zero_to_one_index = np.where(zero_one_sub == -1)[0]
    one_to_zero_index = np.where(zero_one_sub == 1)[0]


    if zero_to_one_index[0] > one_to_zero_index[0]:
        one_to_zero_index = one_to_zero_index[1:]
    if zero_to_one_index.shape[0] > one_to_zero_index.shape[0]:
        zero_to_one_index = zero_to_one_index[:-1]

    list_indexes = list()

    for i,j in zip(zero_to_one_index, one_to_zero_index):
        if j-i >= chunk_size: 
            list_indexes.append([i,j])

    return list_indexes


def getChunks_anchors(labels, game_index, params, chunk_size=240, receptive_field=80):

    # get indexes of labels
    indexes=list()
    for i in np.arange(labels.shape[1]):
        indexes.append(np.where(labels[:,i] == 0)[0].tolist())

    # Positive chunks
    anchors = list()

    class_counter = 0
    for event in indexes:
        for element in event:
            anchors.append([game_index,element,class_counter])
        class_counter += 1


    # Negative chunks

    negative_indexes = getNegativeIndexes(labels, params, chunk_size)

    for negative_index in negative_indexes:
        start = [negative_index[0], negative_index[1]]
        anchors.append([game_index,start,labels.shape[1]])


    return anchors

def getTimestampTargets(labels, num_detections):

    targets = np.zeros((labels.shape[0],num_detections,2+labels.shape[-1]), dtype='float')

    for i in np.arange(labels.shape[0]):

        time_indexes, class_values = np.where(labels[i]==0)

        counter = 0

        for time_index, class_value in zip(time_indexes, class_values):

            # Confidence
            targets[i,counter,0] = 1.0 
            # frame index normalized
            targets[i,counter,1] = time_index/(labels.shape[1])
            # The class one hot encoded
            targets[i,counter,2+class_value] = 1.0
            counter += 1

            if counter >= num_detections:
                print("More timestamp than what was fixed... A lot happened in that chunk")
                break

    return targets



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

        tmp_segmentation = torch.nn.functional.one_hot(torch.argmax(output_segmentation[batch], dim=-1), num_classes=output_segmentation.size()[-1])


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