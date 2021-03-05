
import numpy as np

def oneHotToAlllabels(onehot):
    
    nb_frames = onehot.shape[0]
    nb_camera = onehot.shape[1]

    onehot=np.flip(onehot,0)
    Frames_Camera = onehot
    camera_type=0
    camera_length=nb_frames
    count_shot=0
    for i in range(nb_camera):
        y = onehot[:,i]
        #print('y',y.shape)
        camera_change= np.where(y == 1)[0]
        #print(camera_change.shape)
        if y[camera_change].size>0:
            if camera_length<camera_change[0]:
                camera_length=camera_change[0]
                camera_type=i
    
    #print(onehot.shape,range(nb_frames))       
    for i in range(nb_frames):
        #print(i)
        x = onehot[i,:]
        loc_events = np.where(x == 1)[0]
        nb_events = len(loc_events)
        if x[loc_events].size>0:
            camera_type=loc_events[0]
            count_shot=count_shot+1

        Frames_Camera[i,camera_type] = 1
    #print(count_shot,nb_frames)
    return np.flip(Frames_Camera,0)

def oneClasslabels(onehot):
    
    nb_frames = onehot.shape[0]
    nb_camera = onehot.shape[1]
    
    oneclasslabel = np.empty(nb_frames)
    camera_type=0
    camera_length=nb_frames
            
    for i in range(nb_frames):
        
        x = onehot[i,:]
        loc_events = np.where(x == 1)[0]
        nb_events = len(loc_events)
        if x[loc_events].size>0:
            oneclasslabel[i] = 1

        
    
    return oneclasslabel
def getNegativeIndexes(labels, params, chunk_size):

    zero_one_labels = np.zeros(labels.shape)
    for i in np.arange(labels.shape[1]):
        zero_one_labels[:,i] = 1-np.logical_or(np.where(labels[:,i] >= params[3], 1,0),np.where(labels[:,i] <= params[0], 1,0))
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
        if j-i >= chunk_size: #args.chunksize*args.framerate:
            list_indexes.append([i,j])

    return list_indexes


def getChunks(features, labels, chunk_size=240, receptive_field=80):

    # get indexes of labels
    indexes=list()
    for i in np.arange(labels.shape[1]):
        indexes.append(np.where(labels[:,i] == 0)[0].tolist())

    # Positive chunks
    positives_chunks_features = list()
    positives_chunks_labels = list()

    chunk_size = 240 #args.chunksize*args.framerate
    receptive_field = 80 # args.receptivefield*args.framerate

    for event in indexes:
        for element in event:
            shift = random.randint(-chunk_size+receptive_field, -receptive_field)
            start = element + shift
            if start < 0:
                start = 0
            if start+chunk_size >= features.shape[0]:
                start = features.shape[0]-chunk_size-1
            positives_chunks_features.append(features[start:start+chunk_size])
            positives_chunks_labels.append(labels[start:start+chunk_size])


    # Negative chunks
    # number_of_negative_chunks = np.floor(len(positives_chunks_labels)/labels.shape[1])+1
    # negatives_chunks_features = list()
    # negatives_chunks_labels = list()

    # negative_indexes = getNegativeIndexes(labels, [-20,-10,10,20])

    # counter = 0
    # while counter < number_of_negative_chunks and counter < len(negative_indexes):
    #     selection = random.randint(0, len(negative_indexes)-1)
    #     start = random.randint(negative_indexes[selection][0], negative_indexes[selection][1]-chunk_size)
    #     if start < 0:
    #         start = 0
    #     if start+chunk_size >= features.shape[0]:
    #         start = features.shape[0]-chunk_size-1
    #     negatives_chunks_features.append(features[start:start+chunk_size])
    #     negatives_chunks_labels.append(labels[start:start+chunk_size])
    #     counter += 1

    positives_array_features = np.array(positives_chunks_features)
    positives_array_labels = np.array(positives_chunks_labels)
    # negatives_array_features = np.array(negatives_chunks_features)
    # negatives_array_labels = np.array(negatives_chunks_labels)

    # inputs = None
    # targets = None

    # # print(positives_array_features.shape[0],  negatives_array_features.shape[0])
    # if positives_array_features.shape[0] > 0 and negatives_array_features.shape[0] > 0:
    #     inputs = np.copy(np.concatenate((positives_array_features, negatives_array_features), axis=0))
    #     targets = np.copy(np.concatenate((positives_array_labels, negatives_array_labels), axis=0))
    # elif negatives_array_features.shape[0] == 0:
    #     inputs = np.copy(positives_array_features)
    #     targets = np.copy(positives_array_labels)
    # else:
    #     inputs = np.copy(negatives_array_features)
    #     targets = np.copy(negatives_array_labels)
    # if positives_array_features.shape[0] == 0 and negatives_array_features.shape[0] == 0:
    #     print("No chunks could be retrieved...")
    
    
    # Put loss to zero outside receptive field
    inputs = np.copy(positives_array_features)
    targets = np.copy(positives_array_labels)
    targets[:,0:int(np.ceil(receptive_field/2)),:] = -1
    targets[:,-int(np.ceil(receptive_field/2)):,:] = -1

    return inputs, targets


# def segemtation_to_Onehot(sgementation,num_classes=12):
#     onehot = np.zeros(sgementation.shape[1],num_classes)
#     for in np.arange(sgementation.shape[1]):
#         if sgementation[i,1] in np.arange(num_classes):
#             onehot[i,sgementation[i,1]]=1
        
#     return onehot

def getChunks_anchors(labels, game_index, chunk_size=240, receptive_field=80):

    # get indexes of labels
    indexes=list()
    indexes.append(np.where(labels == 1)[0].tolist())

    # Positive chunks
    positives_chunks_anchors = list()
    negatives_chunks_anchors = list()

    for event in indexes:
        for element in event:
            positives_chunks_anchors.append([game_index,element])


    # Negative chunks
    # number_of_negative_chunks = np.floor(len(positives_chunks_anchors)/labels.shape[1])+1
    # negatives_chunks_features = list()
    # negatives_chunks_labels = list()

    # negative_indexes = getNegativeIndexes(labels, [-20,-10,10,20], chunk_size)

    # counter = 0
    # while counter < number_of_negative_chunks and counter < len(negative_indexes):
    #     selection = random.randint(0, len(negative_indexes)-1)
    #     start = random.randint(negative_indexes[selection][0], negative_indexes[selection][1]-chunk_size)
    #     if start < 0:
    #         start = 0
    #     if start+chunk_size >= labels.shape[0]:
    #         start = labels.shape[0]-chunk_size-1
    #     negatives_chunks_anchors.append([game_index,start+(chunk_size//2)])
    #     counter += 1

    positives_chunks_anchors = np.array(positives_chunks_anchors)
    # negatives_chunks_anchors = np.array(negatives_chunks_anchors)

    anchors = None

    # if positives_chunks_anchors.shape[0] > 0 and negatives_chunks_anchors.shape[0] > 0:
    #     anchors = np.copy(np.concatenate((positives_chunks_anchors, negatives_chunks_anchors), axis=0))
    # elif negatives_chunks_anchors.shape[0] == 0:
    #    
    # else:
    #     anchors = np.copy(negatives_chunks_anchors)
    # if positives_chunks_anchors.shape[0] == 0 and negatives_chunks_anchors.shape[0] == 0:
    #     print("No chunks could be retrieved...")

    anchors = np.copy(positives_chunks_anchors)
    return anchors

def getTimestampTargets(labels, num_detections):
   
    targets = np.zeros((labels.shape[0],num_detections,2), dtype='float')

    for i in np.arange(labels.shape[0]):

        time_indexes, class_values = np.where(labels[i]==1)
        
        counter = 0

        for time_index, class_value in zip(time_indexes, class_values):

            # Confidence
            targets[i,counter,0] = 1.0 
            # frame index normalized
            targets[i,counter,1] = time_index/(labels.shape[1])
            counter += 1

            if counter >= num_detections:
                print("More timestamp than what was fixed... A lot happened in that chunk")
                break
        
    return targets
