import argparse
import os
import SoccerNet

import logging

import configparser
import math
try:
    # pip install tensorflow (==2.3.0)
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.resnet import preprocess_input
    # from tensorflow.keras.preprocessing.image import img_to_array
    # from tensorflow.keras.preprocessing.image import load_img
    from tensorflow import keras
except:
    print("issue loading TF2")
    pass
import os
# import argparse
import numpy as np
import cv2  # pip install opencv-python (==3.4.11.41)
import imutils  # pip install imutils
import skvideo.io
from tqdm import tqdm
import pickle as pkl

from sklearn.decomposition import PCA, IncrementalPCA  # pip install scikit-learn
from sklearn.preprocessing import StandardScaler
import json

import random
from SoccerNet.utils import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.DataLoader import Frame, FrameCV


class VideoFeatureExtractor():
    def __init__(self,
                 feature="ResNET",
                 back_end="TF2",
                 overwrite=False,
                 transform="crop",
                 grabber="opencv",
                 FPS=2.0,
                 split="all"):

        self.feature = feature
        self.back_end = back_end
        self.verbose = True
        self.transform = transform
        self.overwrite = overwrite
        self.grabber = grabber
        self.FPS = FPS
        self.split = split


        if "TF2" in self.back_end:

            # create pretrained encoder (here ResNet152, pre-trained on ImageNet)
            base_model = keras.applications.resnet.ResNet152(include_top=True,
                                                             weights='imagenet',
                                                             input_tensor=None,
                                                             input_shape=None,
                                                             pooling=None,
                                                             classes=1000)

            # define model with output after polling layer (dim=2048)
            self.model = Model(base_model.input,
                               outputs=[base_model.get_layer("avg_pool").output])
            self.model.trainable = False

        
    def extractFeatures(self, path_video_input, path_features_output, start=None, duration=None, overwrite=False):
        logging.info(f"extracting features for video {path_video_input}")

        if os.path.exists(path_features_output) and not overwrite:
            logging.info("Features already exists, use overwrite=True to overwrite them. Exiting.")
            return
        if "TF2" in self.back_end:

            if self.grabber == "skvideo":
                videoLoader = Frame(
                    path_video_input, FPS=self.FPS, transform=self.transform, start=start, duration=duration)
            elif self.grabber == "opencv":
                videoLoader = FrameCV(
                    path_video_input, FPS=self.FPS, transform=self.transform, start=start, duration=duration)

            frames = preprocess_input(videoLoader.frames)

            if duration is None:
                duration = videoLoader.time_second

            logging.info(f"frames {frames.shape}, fps={frames.shape[0]/duration}")

            # predict the features from the frames (adjust batch size for smaller GPU)
            features = self.model.predict(frames, batch_size=64, verbose=1)

            logging.info(f"features {features.shape}, fps={features.shape[0]/duration}")

        # save the featrue in .npy format
        os.makedirs(os.path.dirname(path_features_output), exist_ok=True)
        np.save(path_features_output, features)


class PCAReducer():
    def __init__(self, pca_file=None, scaler_file=None):
        self.pca_file = pca_file
        self.scaler_file = scaler_file
        self.loadPCA()
    
    def loadPCA(self):
        # Read pre-computed PCA
        self.pca = None
        if self.pca_file is not None:
            with open(self.pca_file, "rb") as fobj:
                self.pca = pkl.load(fobj)

        # Read pre-computed average
        self.average = None
        if self.scaler_file is not None:
            with open(self.scaler_file, "rb") as fobj:
                self.average = pkl.load(fobj)

    def reduceFeatures(self, input_features, output_features, overwrite=False):
        logging.info(f"reducing features {input_features}")

        if os.path.exists(output_features) and not overwrite:
            logging.info(
                "Features already exists, use overwrite=True to overwrite them. Exiting.")
            return
        feat = np.load(input_features)
        if self.average is not None:
            feat = feat - self.average
        if self.pca is not None:
            feat = self.pca.transform(feat)
        np.save(output_features, feat)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Extract ResNet feature from a video.')

    parser.add_argument('--path_video', type=str, required=True,
                        help="Path of the Input Video")
    parser.add_argument('--path_features', type=str, required=True,
                        help="Path of the Output Features")
    parser.add_argument('--start', type=float, default=None,
                        help="time of the video to strat extracting features [default:None]")
    parser.add_argument('--duration', type=float, default=None,
                        help="duration of the video before finishing extracting features [default:None]")
    parser.add_argument('--overwrite', action="store_true",
                        help="Overwrite the features.")

    parser.add_argument('--GPU', type=int, default=0,
                        help="ID of the GPU to use [default:0]")
    parser.add_argument('--loglevel', type=str, default="INFO",
                        help="loglevel for logging [default:INFO]")

    # feature setup
    parser.add_argument('--back_end', type=str, default="TF2",
                        help="Backend TF2 or PT [default:TF2]")
    parser.add_argument('--features', type=str, default="ResNET",
                        help="ResNET or R25D [default:ResNET]")
    parser.add_argument('--transform', type=str, default="crop",
                        help="crop or resize? [default:crop]")
    parser.add_argument('--video', type=str, default="LQ",
                        help="LQ or HQ? [default:LQ]")
    parser.add_argument('--grabber', type=str, default="opencv",
                        help="skvideo or opencv? [default:opencv]")
    parser.add_argument('--FPS', type=float, default=2.0,
                        help="FPS for the features [default:2.0]")

    # PCA reduction
    parser.add_argument('--PCA', type=str, default="pca_512_TF2.pkl",
                        help="Pickle with pre-computed PCA")
    parser.add_argument('--PCA_scaler', type=str, default="average_512_TF2.pkl",
                        help="Pickle with pre-computed PCA scaler")
                        
    args = parser.parse_args()
    # print(args)
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), None),
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.StreamHandler()
        ])


    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    myFeatureExtractor = VideoFeatureExtractor(
        feature=args.features,
        back_end=args.back_end,
        transform=args.transform,
        grabber=args.grabber,
        FPS=args.FPS)

    myFeatureExtractor.extractFeatures(path_video_input=args.path_video,
                                       path_features_output=args.path_features,
                                       start=args.start,
                                       duration=args.duration,
                                       overwrite=args.overwrite)

    if args.PCA is not None or args.PCA_scaler is not None:
        myPCAReducer = PCAReducer(pca_file=args.PCA,
                                  scaler_file=args.PCA_scaler)

        myPCAReducer.reduceFeatures(input_features=args.path_features,
                                    output_features=args.path_features,
                                    overwrite=args.overwrite)

# import numpy as np
# np.load("/media/giancos/Football/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/BUTTA.npy") - np.load("/media/giancos/Football/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_ResNET_TF2.npy")
