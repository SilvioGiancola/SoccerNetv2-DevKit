import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import f1_score
np.seterr(divide='ignore', invalid='ignore')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
        
def append_function(targets,detections, num_classes):
    total_detections = np.zeros((1, num_classes))
    total_labels = np.zeros((1, num_classes))
    total_detections[0,0] = -1
    n_gt_labels = 0
   
    for target, detection in zip(targets, detections):
        total_detections = np.append(total_detections, detection, axis=0)
        total_labels = np.append(total_labels, target, axis=0)

    return total_labels,total_detections

def calculate_f1_score(targets, detections):
    num_classes = 13
    total_labels, total_detections = append_function(targets, detections, num_classes)
    # change Onehot to integer 
    groundtruth=np.argmax(total_labels, axis=1)
    int_detections=np.argmax(total_detections, axis=1)
 
    f1_macro=f1_score(groundtruth, int_detections, average='macro')
    f1_micro=f1_score(groundtruth, int_detections, average='micro')
    
    return f1_macro, f1_micro,calculate_f1_manual(groundtruth,int_detections,num_classes)

def calculate_f1_manual(groundtruth,int_detections,num_classes):
    #num_class=14
    f1_s=np.zeros((num_classes,1))
    for i in np.arange(num_classes):
        groundtruth_i=groundtruth[np.where(groundtruth==i)[0]]==i
        int_detections_i=int_detections[np.where(groundtruth==i)[0]]==i
        f1_s[i]=f1_score(groundtruth_i,int_detections_i)
    
    return np.sum(f1_s[0:12])/12

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

def compute_class_scores(target, detection, delta):

    # Retrieving the important variables
    gt_indexes = np.where(target != 0)[0]
    pred_indexes = np.where(detection >= 0)[0]
    pred_scores = detection[pred_indexes]

    # Array to save the results, each is [pred_scor,{1 or 0}]
    game_detections = np.zeros((len(pred_indexes),2))
    game_detections[:,0] = np.copy(pred_scores)


    remove_indexes = list()

    for gt_index in gt_indexes:

        max_score = -1
        max_index = None
        game_index = 0
        selected_game_index = 0

        for pred_index, pred_score in zip(pred_indexes, pred_scores):

            if abs(pred_index-gt_index) <= delta/2 and pred_score > max_score and pred_index not in remove_indexes:
                max_score = pred_score
                max_index = pred_index
                selected_game_index = game_index
            game_index += 1

        if max_index is not None:
            game_detections[selected_game_index,1]=1
            remove_indexes.append(max_index)

    return game_detections, len(gt_indexes)



def compute_precision_recall_curve(targets, detections, delta, NMS_on):
    
    # Store the number of classes
    num_classes = targets[0].shape[-1]

    # 200 confidence thresholds between [0,1]
    thresholds = np.linspace(0,1,200)

    # Store the precision and recall points
    precision = list()
    recall = list()

    # Apply Non-Maxima Suppression if required
    start = time.time()
    detections_NMS = list()
    if NMS_on:
        for detection in detections:
            detections_NMS.append(NMS(detection,delta))
    else:
        detections_NMS = detections

    # Precompute the predictions scores and their correspondence {TP, FP} for each class
    for c in np.arange(num_classes):
        total_detections =  np.zeros((1, 2))
        total_detections[0,0] = -1
        n_gt_labels = 0
        
        # Get the confidence scores and their corresponding TP or FP characteristics for each game
        for target, detection in zip(targets, detections_NMS):
            tmp_detections, tmp_n_gt_labels = compute_class_scores(target[:,c], detection[:,c], delta)
            total_detections = np.append(total_detections,tmp_detections,axis=0)
            n_gt_labels = n_gt_labels + tmp_n_gt_labels

        precision.append(list())
        recall.append(list())

        # Get the precision and recall for each confidence threshold
        for threshold in thresholds:
            pred_indexes = np.where(total_detections[:,0]>=threshold)[0]
            TP = np.sum(total_detections[pred_indexes,1])
            p = np.nan_to_num(TP/len(pred_indexes))
            r = np.nan_to_num(TP/n_gt_labels)
            precision[-1].append(p)
            recall[-1].append(r)

    precision = np.array(precision).transpose()
    recall = np.array(recall).transpose()


    # Sort the points based on the recall, class per class
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall[:,i])
        precision[:,i] = precision[index_sort,i]
        recall[:,i] = recall[index_sort,i]

    return precision, recall

def compute_mAP(precision, recall):

    # Array for storing the AP per class
    AP = np.array([0.0]*precision.shape[-1])

    # Loop for all classes
    for i in np.arange(precision.shape[-1]):

        # 11 point interpolation
        for j in np.arange(11)/10:

            index_recall = np.where(recall[:,i] >= j)[0]

            possible_value_precision = precision[index_recall,i]
            max_value_precision = 0

            if possible_value_precision.shape[0] != 0:
                max_value_precision = np.max(possible_value_precision)

            AP[i] += max_value_precision

    mAP_per_class = AP/11

    return np.mean(mAP_per_class)

def delta_curve(targets, detections,  framerate, NMS_on):

    mAP = list()

    for delta in tqdm((np.arange(1)*1 + 1)*framerate):

        precision, recall = compute_precision_recall_curve(targets, detections, delta, NMS_on)

        mAP.append(compute_mAP(precision, recall))
        # plt.plot(precision, recall, 'ro')
        # plt.savefig('/content/camerashot.png')

    return mAP


def average_mAP(targets, detections, framerate=2, NMS_on=True):

    targets_numpy = list()
    detections_numpy = list()
    
    for target, detection in zip(targets,detections):
        targets_numpy.append(target)
        detections_numpy.append(detection)

    mAP = delta_curve(targets_numpy, detections_numpy, framerate, NMS_on)
    
    #Compute the average mAP
#     integral = 0.0
#     for i in np.arange(len(mAP)-1):
#         integral += 5*(mAP[i]+mAP[i+1])/2
#     a_mAP = integral/(5*(len(mAP)-1))
    return mAP
