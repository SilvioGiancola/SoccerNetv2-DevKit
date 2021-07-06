import os
import json
import numpy as np
from SoccerNet.Downloader import getListGames
from config.classes import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2

def predictions2json(predictions_half_1, output_path, framerate=2):

    os.makedirs(output_path, exist_ok=True)
    output_file_path = output_path + "/Predictions-v2.json"

    frames_half_1, class_half_1 = np.where(predictions_half_1 >= 0)

    json_data = dict()
    json_data["predictions"] = list()

    for frame_index, class_index in zip(frames_half_1, class_half_1):

        confidence = predictions_half_1[frame_index, class_index]

        seconds = int((frame_index//framerate)%60)
        minutes = int((frame_index//framerate)//60)

        prediction_data = dict()
        prediction_data["gameTime"] = str(1) + " - " + str(minutes) + ":" + str(seconds)
        prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[class_index]
        prediction_data["position"] = str(int((frame_index/framerate)*1000))
        prediction_data["half"] = str(1)
        prediction_data["confidence"] = str(confidence)

        json_data["predictions"].append(prediction_data)

    with open(output_file_path, 'w') as output_file:
        json.dump(json_data, output_file, indent=4)
