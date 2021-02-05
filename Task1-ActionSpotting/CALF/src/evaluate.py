import os
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from json_io import label2vector, predictions2vector
from SoccerNet.Downloader import getListGames
from metrics_visibility_fast import average_mAP

def evaluate_average_mAP(SoccerNet_path, Predictions_path, Evaluation_set, framerate):
    list_games = getListGames(Evaluation_set)
    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()

    for game in list_games:
        label_half_1, label_half_2 = label2vector(SoccerNet_path + game)
        predictions_half_1, predictions_half_2 = predictions2vector(SoccerNet_path + game, Predictions_path + game)

        targets_numpy.append(label_half_1)
        targets_numpy.append(label_half_2)
        detections_numpy.append(predictions_half_1)
        detections_numpy.append(predictions_half_2)

        closest_numpy = np.zeros(label_half_1.shape)-1
        #Get the closest action index
        for c in np.arange(label_half_1.shape[-1]):
            indexes = np.where(label_half_1[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = label_half_1[indexes[i],c]
        closests_numpy.append(closest_numpy)

        closest_numpy = np.zeros(label_half_2.shape)-1
        for c in np.arange(label_half_2.shape[-1]):
            indexes = np.where(label_half_2[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = label_half_2[indexes[i],c]
        closests_numpy.append(closest_numpy)


    # Compute the performances
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = average_mAP(targets_numpy, detections_numpy, closests_numpy, framerate)
    
    print("Average mAP: ", a_mAP)
    print("Average mAP visible: ", a_mAP_visible)
    print("Average mAP unshown: ", a_mAP_unshown)
    print("Average mAP per class: ", a_mAP_per_class)
    print("Average mAP visible per class: ", a_mAP_per_class_visible)
    print("Average mAP unshown per class: ", a_mAP_per_class_unshown)

    return a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown 

if __name__ == '__main__':

    # Load the arguments
    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--SoccerNet_path',   required=True, type=str, help='Path to the SoccerNet-V2 dataset folder' )
    parser.add_argument('--Predictions_path',   required=True, type=str, help='Path to the predictions folder' )
    parser.add_argument('--Evaluation_set',   required=False, type=str, default= "test", help='Set on which to evaluate the performances' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )

    args = parser.parse_args()



    evaluate_average_mAP(args.SoccerNet_path, args.Predictions_path, args.Evaluation_set, args.framerate)
