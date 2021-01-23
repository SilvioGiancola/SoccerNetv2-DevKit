import numpy as np
from tqdm import tqdm
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



def confusion_matrix_single_game(targets, detections, delta, threshold):

    # Get all targets indexes for each class
    num_classes = targets.shape[1]

    TP = list()
    FP = list()
    FN = list()

    # Iterate over all classes
    for i in np.arange(num_classes):
        gt_indexes = np.where(targets[:,i]==1)[0]
        pred_indexes = np.where(detections[:,i] >=threshold)[0]
        pred_scores = detections[pred_indexes,i]

        # If there are no groundtruths
        if len(gt_indexes) == 0:
            TP.append(0)
            FP.append(len(pred_indexes))
            FN.append(0)
            continue

        # If there are no predictions
        if len(pred_indexes) == 0:
            TP.append(0)
            FP.append(0)
            FN.append(len(gt_indexes))
            continue

        # Iterate over all groundtruths
        TP_class = 0
        FP_class = 0
        FN_class = 0
        remove_indexes = list()

        for gt_index in gt_indexes:
            # Get the predictions which are within the delta interval of each 
            max_score = -1
            max_index = None
            for pred_index, pred_score in zip(pred_indexes, pred_scores):
                # The two indexes are very close to each other, choose the one with the greatest score
                if abs(pred_index-gt_index) <= delta/2 and pred_score > max_score and pred_index not in remove_indexes:
                    max_score = pred_score
                    max_index = pred_index
            # If, for this groundtruth, no predictions could fit
            if max_index is None:
                FN_class += 1
            # If there is one good prediction
            else:
                TP_class += 1
                remove_indexes.append(max_index)
        
        FP_class = len(pred_indexes)-len(remove_indexes)

        TP.append(TP_class)
        FP.append(FP_class)
        FN.append(FN_class)

    return TP, FP, FN


def compute_confusion_matrix(targets, detections, delta, threshold):

    TP = np.array([0]*targets[0].shape[1])
    FP = np.array([0]*targets[0].shape[1])
    FN = np.array([0]*targets[0].shape[1])

    for target, detection in zip(targets, detections):
        TP_tmp, FP_tmp, FN_tmp = confusion_matrix_single_game(target, detection, delta, threshold)
        TP += np.array(TP_tmp)
        FP += np.array(FP_tmp)
        FN += np.array(FN_tmp)

    return TP, FP, FN

def compute_precision_recall_curve(targets, detections, delta, NMS_on):

    # 200 confidence thresholds between [0,1]
    thresholds = np.linspace(0,1,200)

    # Store the precision and recall points
    precision = list()
    recall = list()

    # Apply Non-Maxima Suppression if required
    detections_NMS = list()
    if NMS_on:
        for detection in detections:
            detections_NMS.append(NMS(detection,delta))
    else:
        detections_NMS = detections

    # Get the precision and recall for each confidence threshold
    for threshold in thresholds:
        TP, FP, FN = compute_confusion_matrix(targets, detections_NMS, delta, threshold)
        p = np.nan_to_num(TP/(TP+FP))
        r = np.nan_to_num(TP/(TP+FN))

        precision.append(p)
        recall.append(r)
    precision = np.array(precision)
    recall = np.array(recall)

    # Sort the points based on the recall, class per class
    for i in np.arange(precision.shape[1]):
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

def delta_curve(targets, detections,  framerate, savepath, NMS_on):

    mAP = list()

    for delta in tqdm((np.arange(12)*5 + 5)*framerate):

        precision, recall = compute_precision_recall_curve(targets, detections, delta, NMS_on)

        mAP.append(compute_mAP(precision, recall))

    return mAP


def average_mAP(targets, detections, framerate=2, savepath=None, NMS_on=True):

    targets_numpy = list()
    detections_numpy = list()
    
    for target, detection in zip(targets,detections):
        targets_numpy.append(target.numpy())
        detections_numpy.append(detection.numpy())

    mAP = delta_curve(targets_numpy, detections_numpy, framerate, savepath, NMS_on)
    # Compute the average mAP
    integral = 0.0
    for i in np.arange(len(mAP)-1):
        integral += 5*(mAP[i]+mAP[i+1])/2
    a_mAP = integral/(5*(len(mAP)-1))

    return a_mAP