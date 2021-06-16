import os
import json
import numpy as np
from SoccerNet.Downloader import getListGames
from config.classes import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2, EVENT_DICTIONARY_V2_VISUAL, INVERSE_EVENT_DICTIONARY_V2_VISUAL, EVENT_DICTIONARY_V2_NONVISUAL, INVERSE_EVENT_DICTIONARY_V2_NONVISUAL

def label2vector(folder_path, num_classes=17, framerate=2):

    label_path = folder_path + "/Labels-v2.json"

    # Load labels
    labels = json.load(open(label_path))

    vector_size = 90*60*framerate

    label_half1 = np.zeros((vector_size, num_classes))
    label_half2 = np.zeros((vector_size, num_classes))

    for annotation in labels["annotations"]:

        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        frame = framerate * ( seconds + 60 * minutes ) 

        if event not in EVENT_DICTIONARY_V2:
            continue
        label = EVENT_DICTIONARY_V2[event]

        value = 1
        if "visibility" in annotation.keys():
            if annotation["visibility"] == "not shown":
                value = -1

        if half == 1:
            frame = min(frame, vector_size-1)
            label_half1[frame][label] = value

        if half == 2:
            frame = min(frame, vector_size-1)
            label_half2[frame][label] = value

    return label_half1, label_half2

def predictions2json(predictions_half_1, predictions_half_2, output_path, game_info, framerate=2, split_class=None):

    os.makedirs(output_path + game_info, exist_ok=True)
    output_file_path = output_path + game_info + "/Predictions-v2.json"

    frames_half_1, class_half_1 = np.where(predictions_half_1 >= 0)
    frames_half_2, class_half_2 = np.where(predictions_half_2 >= 0)

    json_data = dict()
    json_data["UrlLocal"] = game_info
    json_data["predictions"] = list()

    INVERSE_DICT = INVERSE_EVENT_DICTIONARY_V2
    if split_class == "visual":
        INVERSE_DICT = INVERSE_EVENT_DICTIONARY_V2_VISUAL
    elif split_class == "nonvisual":
        INVERSE_DICT = INVERSE_EVENT_DICTIONARY_V2_NONVISUAL

    for frame_index, class_index in zip(frames_half_1, class_half_1):

        confidence = predictions_half_1[frame_index, class_index]

        seconds = int((frame_index//framerate)%60)
        minutes = int((frame_index//framerate)//60)

        prediction_data = dict()
        prediction_data["gameTime"] = str(1) + " - " + str(minutes) + ":" + str(seconds)
        prediction_data["label"] = INVERSE_DICT[class_index]
        prediction_data["position"] = str(int((frame_index/framerate)*1000))
        prediction_data["half"] = str(1)
        prediction_data["confidence"] = str(confidence)

        json_data["predictions"].append(prediction_data)

    for frame_index, class_index in zip(frames_half_2, class_half_2):

        confidence = predictions_half_2[frame_index, class_index]

        seconds = int((frame_index//framerate)%60)
        minutes = int((frame_index//framerate)//60)

        prediction_data = dict()
        prediction_data["gameTime"] = str(2) + " - " + str(minutes) + ":" + str(seconds)
        prediction_data["label"] = INVERSE_DICT[class_index]
        prediction_data["position"] = str(int((frame_index/framerate)*1000))
        prediction_data["half"] = str(2)
        prediction_data["confidence"] = str(confidence)

        json_data["predictions"].append(prediction_data)

    with open(output_file_path, 'w') as output_file:
        json.dump(json_data, output_file, indent=4)

def predictions2vector(folder_path, predictions_path, num_classes=17, framerate=2):

    predictions_path = predictions_path + "/Predictions-v2.json"

    # Load features

    # Load labels
    predictions = json.load(open(predictions_path))

    vector_size = 90*60*framerate

    prediction_half1 = np.zeros((vector_size, num_classes))-1
    prediction_half2 = np.zeros((vector_size, num_classes))-1

    for annotation in predictions["predictions"]:

        time = int(annotation["position"])
        event = annotation["label"]

        half = int(annotation["half"])

        frame = int(framerate * ( time/1000 ))

        if event not in EVENT_DICTIONARY_V2:
            continue
        label = EVENT_DICTIONARY_V2[event]

        value = annotation["confidence"]

        if half == 1:
            frame = min(frame, vector_size-1)
            prediction_half1[frame][label] = value

        if half == 2:
            frame = min(frame, vector_size-1)
            prediction_half2[frame][label] = value

    return prediction_half1, prediction_half2
