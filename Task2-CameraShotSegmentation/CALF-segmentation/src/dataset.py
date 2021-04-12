from torch.utils.data import Dataset

import numpy as np
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
from config.classes import EVENT_DICTIONARY_V1, EVENT_DICTIONARY_V2, K_V1, K_V2,Camera_Change_DICTIONARY,Camera_Type_DICTIONARY

from preprocessing import oneHotToAlllabels, getTimestampTargets, getChunks, getChunks_anchors



class SoccerNetClips(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="train", version=1, framerate=2, chunk_size=240, receptive_field=80,num_detections=45):
        self.path = path
        self.listGames = getListGames(split, task="camera-changes")
        #self.listGames = np.load(os.path.join(self.path, split))
        self.features = features
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.framerate=framerate
        self.version = version

        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
            self.K_parameters = K_V1*framerate
            self.num_detections =5
        elif version == 2:
            self.dict_change = Camera_Change_DICTIONARY
            self.dict_type = Camera_Type_DICTIONARY
            self.num_classes_sgementation = 13
            self.num_classes_camera_change = 1
            self.labels="Labels-cameras.json"
            self.K_parameters = K_V2*framerate  
            self.num_detections =num_detections


        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False, task="camera-changes", randomized=True)


        logging.info("Pre-compute clips")

        clip_feats = []
        clip_labels = []

        self.game_feats = list()
        self.game_labels = list()
        self.game_change_labels = list()
        self.game_anchors = list()
        game_counter = 0
        for game in tqdm(self.listGames):
            # Load features
            #10% of dataset
            # if np.random.randint(10, size=1)<8:
            #      continue
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))

            # Load labels
            labels = json.load(open(os.path.join(path, game, self.labels)))

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes_sgementation))
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes_sgementation))
            label_change_half1 = np.zeros((feat_half1.shape[0], self.num_classes_camera_change))
            label_change_half2 = np.zeros((feat_half2.shape[0], self.num_classes_camera_change))

            for annotation in labels["annotations"]:
                time = annotation["gameTime"]
                camera_type = annotation["label"]
                camera_change=annotation["change_type"]
                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                framerate=self.framerate
                frame = framerate * ( seconds + 60 * minutes )

                if camera_type  in self.dict_type:
                
                    label_type = self.dict_type[camera_type]

                    if half == 1:
                        frame = min(frame, feat_half1.shape[0]-1)
                        label_half1[frame][label_type] = 1

                    if half == 2:
                        frame = min(frame, feat_half2.shape[0]-1)
                        label_half2[frame][label_type] = 1

                #Onehot for camera changee
                if camera_change in self.dict_change:
                
                    label_change = self.dict_change[camera_change]

                    if half == 1:
                        frame = min(frame, feat_half1.shape[0]-1)
                        label_change_half1[frame] = 1

                    if half == 2:
                        frame = min(frame, feat_half2.shape[0]-1)
                        label_change_half2[frame] = 1
            shift_half1 = oneHotToAlllabels(label_half1)
            shift_half2 = oneHotToAlllabels(label_half2)

            #anchors_half1 = getChunks_anchors(shift_half1, game_counter, self.chunk_size, self.receptive_field)
            anchors_half1 = getChunks_anchors(label_change_half1, game_counter, self.chunk_size, self.receptive_field)         
            game_counter = game_counter+1
            # with np.printoptions(threshold=np.inf):
            #     print('anchors_half1',anchors_half1)
            #anchors_half2 = getChunks_anchors(shift_half2, game_counter, self.chunk_size, self.receptive_field)
            anchors_half2 = getChunks_anchors(label_change_half2, game_counter, self.chunk_size, self.receptive_field)

            game_counter = game_counter+1


            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(shift_half1)
            self.game_labels.append(shift_half2)
            # with np.printoptions(threshold=np.inf):
            #     print('game_labels',self.game_labels)
            self.game_change_labels.append(label_change_half1)
            self.game_change_labels.append(label_change_half2)
            # with np.printoptions(threshold=np.inf):
            #     print('game_change_labels',self.game_change_labels)
            for anchor in anchors_half1:
                self.game_anchors.append(anchor)
            for anchor in anchors_half2:
                self.game_anchors.append(anchor)



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
        """

        # Retrieve the game index and the anchor
        game_index = self.game_anchors[index][0]
        anchor = self.game_anchors[index][1]

        # Compute the shift
        shift = np.random.randint(-self.chunk_size+self.receptive_field, -self.receptive_field)
        start = anchor + shift
        if start < 0:
            start = 0
        if start+self.chunk_size >= self.game_feats[game_index].shape[0]:
            start = self.game_feats[game_index].shape[0]-self.chunk_size-1

        # Extract the clips
        clip_feat = self.game_feats[game_index][start:start+self.chunk_size]
        clip_labels = self.game_labels[game_index][start:start+self.chunk_size]
        clip_change_labels = self.game_change_labels[game_index][start:start+self.chunk_size]

        # # Put loss to zero outside receptive field
        # clip_labels[0:int(np.ceil(self.receptive_field/2)),:] = -1
        # clip_labels[-int(np.ceil(self.receptive_field/2)):,:] = -1

        # clip_change_labels[0:int(np.ceil(self.receptive_field/2)),:] = -1
        # clip_change_labels[-int(np.ceil(self.receptive_field/2)):,:] = -1

        # Get the spotting target
        #print(clip_change_labels.shape)
        clip_targets = getTimestampTargets(np.array([clip_change_labels]), self.num_detections)[0]
        #clip_targets =clip_change_labels
        #print(clip_targets.shape)
        return torch.from_numpy(clip_feat.copy()), torch.from_numpy(clip_labels.copy()), torch.from_numpy(clip_targets.copy())

    def __len__(self):
        return len(self.game_anchors)


class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="test", version=1, framerate=2, chunk_size=240, receptive_field=80,num_detections=45, advanced_test="abrupt"):
        self.path = path
        #self.listGames = getListGames(split)os.path.join(self.dataset_path + self.datatype)
        self.listGames = getListGames(split, task="camera-changes")
        # self.listGames = np.load(os.path.join(self.path, split))
        self.features = features
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.framerate=framerate
        self.version = version
        self.advanced_test = advanced_test
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
            self.K_parameters = K_V1*framerate
            self.num_detections =5
        elif version == 2:
            self.dict_change = Camera_Change_DICTIONARY
            self.dict_type = Camera_Type_DICTIONARY
            self.num_classes_sgementation = 13
            self.num_classes_camera_change = 1
            self.labels="Labels-cameras.json"
            self.K_parameters = K_V2*framerate  
            self.num_detections =num_detections

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False, task="camera-changes", randomized=True)

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
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))

        # Load labels
        labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))

        
        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes_sgementation))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes_sgementation))
        label_change_half1 = np.zeros((feat_half1.shape[0], self.num_classes_camera_change))
        label_change_half2 = np.zeros((feat_half2.shape[0], self.num_classes_camera_change))

        for annotation in labels["annotations"]:

            time = annotation["gameTime"]
            camera_type = annotation["label"]
            camera_change=annotation["change_type"]
            half = int(time[0])

            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            framerate=self.framerate
            frame = framerate * ( seconds + 60 * minutes )-1

            if camera_type  in self.dict_type:
                
                label_type = self.dict_type[camera_type]

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label_type] = 1

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label_type] = 1

            #Onehot for camera changee
            if camera_change in self.dict_change:
                
                label_change = self.dict_change[camera_change]


                value = 1
                if "change_type" in annotation.keys():
                    # if annotation["change_type"] != "logo":
                    if annotation["change_type"] != self.advanced_test:
                    # if annotation["change_type"] != "abrupt":
                        value = -1

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_change_half1[frame] = value

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_change_half2[frame] = value



        label_half1 = oneHotToAlllabels(label_half1)
        label_half2 = oneHotToAlllabels(label_half2)
        def feats2clip(feats, stride, clip_length, padding = "replicate_last"):

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
            for i in torch.arange(0, clip_length):
                idxs.append(idx+i)
            idx = torch.stack(idxs, dim=1)

            if padding=="replicate_last":
                idx = idx.clamp(0, feats.shape[0]-1)

            return feats[idx,:]
            

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)
        feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)
                                  
        return feat_half1, feat_half2, torch.from_numpy(label_change_half1.copy()), torch.from_numpy(label_change_half2.copy()),torch.from_numpy(label_half1.copy()), torch.from_numpy(label_half2.copy())
        # return feat_half1, feat_half2, torch.from_numpy(label_change_half1), torch.from_numpy(label_change_half2)
    def __len__(self):
        return len(self.listGames)



class SoccerNet(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="train", version=1, framerate=2):
        self.path = path
        self.listGames = getListGames(split, task="camera-changes")
        self.features = features
        self.framerate=framerate
        self.version = version
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
            self.K_parameters = K_V1*framerate
            self.num_detections =5
        elif version == 2:
            self.dict_change = Camera_Change_DICTIONARY
            self.dict_shot = Camera_Type_DICTIONARY
            self.num_classes_sgementation = 13
            self.num_classes_camera_change = 1
            self.labels="Labels-cameras.json"
            self.K_parameters = K_V2*framerate  
            self.num_detections =num_detections

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
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))

        # Load labels
        labels = json.load(open(os.path.join(path, self.listGames[index], self.labels)))

        
        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes_sgementation))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes_sgementation))
        label_change_half1 = np.zeros((feat_half1.shape[0], self.num_classes_camera_change))
        label_change_half2 = np.zeros((feat_half2.shape[0], self.num_classes_camera_change))

        for annotation in labels["annotations"]:

            time = annotation["gameTime"]
            camera_type = annotation["label"]
            camera_change=annotation["change_type"]
            half = int(time[0])

            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            framerate=self.framerate
            frame = framerate * ( seconds + 60 * minutes )-1

            if camera_type  in self.dict_type:
                
                label_type = self.dict_type[camera_type]

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label_type] = 1

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label_type] = 1

            #Onehot for camera changee 
            if camera_change in self.dict_change:
                
                label_change = self.dict_change[camera_change]

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_change_half1[frame] = 1

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_change_half2[frame] = 1

        #return feat_half1, feat_half2, torch.from_numpy(label_half1), torch.from_numpy(label_half2), torch.from_numpy(label_change_half1), torch.from_numpy(label_change_half2)
        return feat_half1, feat_half2, torch.from_numpy(label_change_half1.copy()), torch.from_numpy(label_change_half2.copy())
    def __len__(self):
        return len(self.listGames)

        

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    # dataset_Train = SoccerNetClips(path="path/to/SoccerNet/" ,features="ResNET_PCA512.npy", split="train")
    # print(len(dataset_Train))
    # feats, labels = dataset_Train[0]
    # print(feats.shape)
    # print(labels.shape)


    # train_loader = torch.utils.data.DataLoader(dataset_Train,
    #     batch_size=8, shuffle=True,
    #     num_workers=4, pin_memory=True)
    # for i, (feats, labels) in enumerate(train_loader):
    #     print(i, feats.shape, labels.shape)
    # dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split="test", version=args.version, framerate=args.framerate, chunk_size=args.chunk_size*args.framerate, receptive_field=args.receptive_field*args.framerate)

    dataset_Test = SoccerNetClipsTesting(path="path/to/SoccerNet/" ,features="ResNET_PCA512.npy", split="test", version=2)
    print(len(dataset_Test))
    feats1, feats2, labels1, labels2,labels1_2, labels2_2 = dataset_Test[0]
    print(feats1.shape)
    print(labels1.shape)
    print(feats2.shape)
    print(labels2.shape)
    print(feats1[-1])
