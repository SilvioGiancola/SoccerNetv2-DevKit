from torch.utils.data import Dataset

import numpy as np
import random
import os
import time
import ffmpy


from tqdm import tqdm

import torch

import logging
import json

from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1
from Features.VideoFeatureExtractor import VideoFeatureExtractor


def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0):
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
    # print(idx)
    return feats[idx,...]


class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split=["test"], version=1, 
                framerate=2, window_size=15):
        self.path = path
        self.features = features
        self.window_size_frame = window_size*framerate
        self.framerate = framerate
        self.num_classes = 17
        self.num_detections = 15
        self.version = 2

        #Changing video format to 
        ff = ffmpy.FFmpeg(
             inputs={self.path: ""},
             outputs={"inference/outputs/videoLQ.mkv": '-y -r 25 -vf scale=-1:224 -max_muxing_queue_size 9999'})
        print(ff.cmd)
        ff.run()

        print("Initializing feature extractor")
        myFeatureExtractor = VideoFeatureExtractor(
            feature="ResNET",
            back_end="TF2",
            transform="crop",
            grabber="opencv",
            FPS=self.framerate)

        print("Extracting frames")
        myFeatureExtractor.extractFeatures(path_video_input="inference/outputs/videoLQ.mkv",
                                           path_features_output="inference/outputs/features.npy",
                                           overwrite=True)


    def __getitem__(self, index):
        
        # Load features
        feat_half1 = np.load(os.path.join("inference/outputs/features.npy"))
        print("Shape half 1: ", feat_half1.shape)
        size = feat_half1.shape[0]            

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)
                                  
        return feat_half1, size

    def __len__(self):
        return 1

