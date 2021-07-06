import logging
import os
import time
from tqdm import tqdm
import torch
import numpy as np
import math
from preprocessing import batch2long, timestamps2long, visualize, NMS
from json_io import predictions2json

def test(dataloader,model, model_name, save_predictions=False):

    spotting_predictions = list()
    segmentation_predictions = list()

    chunk_size = model.chunk_size
    receptive_field = model.receptive_field

    model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (feat_half1, size) in t:

            feat_half1 = feat_half1.cuda().squeeze(0)


            feat_half1=feat_half1.unsqueeze(1)

            # Compute the output
            output_segmentation_half_1, output_spotting_half_1 = model(feat_half1)


            timestamp_long_half_1 = timestamps2long(output_spotting_half_1.cpu().detach(), size, chunk_size, receptive_field)
            segmentation_long_half_1 = batch2long(output_segmentation_half_1.cpu().detach(), size, chunk_size, receptive_field)

            spotting_predictions.append(timestamp_long_half_1)
            segmentation_predictions.append(segmentation_long_half_1)


    # Transformation to numpy for evaluation
    detections_numpy = list()
    segmentation_numpy = list()
    for segmentation, detection in zip(segmentation_predictions,spotting_predictions):
        segmentation_numpy.append(segmentation.numpy())
        detections_numpy.append(NMS(detection.numpy(), 20*model.framerate))


    # Save the predictions to the json format
    predictions2json(detections_numpy[0],"inference/outputs/", model.framerate)

    #Save the predictions
    for i in np.arange(17):
        visualize(detections_numpy, segmentation_numpy,i)
