from torch.utils.data import Dataset

import numpy as np
import random
# import pandas as pd
import os
import time


from tqdm import tqdm
# import utils

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1



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
    # for i in torch.arange(0, clip_length):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
        # Not replicate last, but take the clip closest to the end of the video
        # idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length
    # print(idx)
    return feats[idx,...]

# def feats2clip(feats, stride, clip_length, padding = "replicate_last"):
#     if padding =="zeropad":
#         print("beforepadding", feats.shape)
#         pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
#         print("pad need to be", clip_length-pad)
#         m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
#         feats = m(feats)
#         print("afterpadding", feats.shape)
#         # nn.ZeroPad2d(2)

#     idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
#     idxs = []
#     for i in torch.arange(0, clip_length):
#         idxs.append(idx+i)
#     idx = torch.stack(idxs, dim=1)

#     if padding=="replicate_last":
#         idx = idx.clamp(0, feats.shape[0]-1)
#         # Not replicate last, but take the clip closest to the end of the video
#         idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length

#     return feats[idx,:]

class SoccerNetClips(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split=["train"], version=1, 
                framerate=2, chunk_size=240):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.chunk_size = chunk_size
        self.version = version
        if version == 1:
            self.num_classes = 3
            self.labels="Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False)


        logging.info("Pre-compute clips")

        self.game_feats = list()
        self.game_labels = list()

        # game_counter = 0
        for game in tqdm(self.listGames):
            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
            feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])
            # print("feat_half1.shape",feat_half1.shape)

            feat_half1 = feats2clip(torch.from_numpy(feat_half1), stride=self.chunk_size, clip_length=self.chunk_size)
            feat_half2 = feats2clip(torch.from_numpy(feat_half2), stride=self.chunk_size, clip_length=self.chunk_size)

            # print("feat_half1.shape",feat_half1.shape)
            # Load labels
            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes+1))
            label_half1[:,0]=1 # those are BG classes
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes+1))
            label_half2[:,0]=1 # those are BG classes


            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * ( seconds + 60 * minutes ) 

                if version == 1:
                    if "card" in event: label = 0
                    elif "subs" in event: label = 1
                    elif "soccer" in event: label = 2
                    else: continue
                elif version == 2:
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]

                # if label outside temporal of view
                if half == 1 and frame//self.chunk_size>=label_half1.shape[0]:
                    continue
                if half == 2 and frame//self.chunk_size>=label_half2.shape[0]:
                    continue

                if half == 1:
                    label_half1[frame//self.chunk_size][0] = 0 # not BG anymore
                    label_half1[frame//self.chunk_size][label+1] = 1 # that's my class

                if half == 2:
                    label_half2[frame//self.chunk_size][0] = 0 # not BG anymore
                    label_half2[frame//self.chunk_size][label+1] = 1 # that's my class
            
            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)

        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        return self.game_feats[index,:,:], self.game_labels[index,:]

    def __len__(self):
        return len(self.game_feats)


class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split=["test"], version=1, 
                framerate=2, chunk_size=240):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.chunk_size = chunk_size
        self.framerate = framerate
        self.version = version
        self.split = split
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        for s in split:
            if s == "challenge":
                downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False)
            else:
                downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        # Load features
        feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
        feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1]) #for C3D non PCA
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))
        feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1]) #for C3D non PCA

        # Load labels
        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))
        
        # check if annoation exists
        if os.path.exists(os.path.join(self.path, self.listGames[index], self.labels)):
        
            labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))
            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = self.framerate * ( seconds + 60 * minutes ) 

                if self.version == 1:
                    if "card" in event: label = 0
                    elif "subs" in event: label = 1
                    elif "soccer" in event: label = 2
                    else: continue
                elif self.version == 2:
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label] = value

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label] = value

            
                

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=1, off=int(self.chunk_size/2), 
                        clip_length=self.chunk_size)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                        stride=1, off=int(self.chunk_size/2), 
                        clip_length=self.chunk_size)

        
        return self.listGames[index], feat_half1, feat_half2, label_half1, label_half2

    def __len__(self):
        return len(self.listGames)



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    dataset_Train = SoccerNetClips(path="/path/to/SoccerNet/" ,features="ResNET_PCA512.npy", split="train")
    print(len(dataset_Train))
    all_labels = []
    from tqdm import tqdm
    for i in tqdm(range(len(dataset_Train))):
        feats, labels = dataset_Train[i]
        all_labels.append(labels)
    all_labels = np.stack(all_labels)
    print(all_labels.shape)
    print(np.sum(all_labels,axis=0))
    print(np.sum(all_labels,axis=1))
        # print(feats.shape, labels)



    # train_loader = torch.utils.data.DataLoader(dataset_Train,
    #     batch_size=8, shuffle=True,
    #     num_workers=4, pin_memory=True)
    # for i, (feats, labels) in enumerate(train_loader):
    #     print(i, feats.shape, labels.shape)

    # dataset_Test = SoccerNetClipsTesting(path="/path/to/SoccerNet/" ,features="ResNET_PCA512.npy")
    # print(len(dataset_Test))
    # for i in range(2):
    #     feats1, feats2, labels1, labels2 = dataset_Test[i]
    #     print(feats1.shape)
    #     print(labels1.shape)
    #     print(feats2.shape)
    #     print(labels2.shape)
    #     # print(feats1[-1])
