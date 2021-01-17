
import argparse

import numpy as np
import os 
from sklearn.decomposition import PCA , IncrementalPCA  # pip install scikit-learn
from sklearn.preprocessing import StandardScaler
from SoccerNet.utils import getListGames
from datetime import datetime
import pickle as pkl

from tqdm import tqdm



def main(args):

    if not os.path.exists(args.pca_file) or not os.path.exists(args.scaler_file):
            
        PCAdata = []
        for game in tqdm(getListGames("v1")):

            half1 = np.load(os.path.join(args.soccernet_dirpath, game, "1_"+args.features))
            PCAdata.append(half1)
            half2 = np.load(os.path.join(args.soccernet_dirpath, game, "2_"+args.features))
            PCAdata.append(half2)

        # Remove average of features
        PCAdata = np.vstack(PCAdata)
        average = np.mean(PCAdata, axis=0)
        PCAdata = PCAdata - average

        # Store average for later
        with open(args.scaler_file, "wb") as fobj:
            pkl.dump(average, fobj)
        

        # Create PCA instance with svd_solver='full' and fit the data
        pca = PCA(n_components=args.dim_reduction, svd_solver='full')
        print(datetime.now(), "PCA start")
        pca.fit(PCAdata)
        print(datetime.now(), "PCA fitted")

        # Store PCA for later
        with open(args.pca_file, "wb") as fobj:
            pkl.dump(pca, fobj)




    # Read pre-computed PCA
    with open(args.pca_file, "rb") as fobj:
        pca = pkl.load(fobj)

    # Read pre-computed average
    with open(args.scaler_file, "rb") as fobj:
        average = pkl.load(fobj)


    # loop over games in v1
    for game in tqdm(getListGames(["v1"])):
        for half in [1,2]:
            game_feat = os.path.join(args.soccernet_dirpath, game, f"{half}_{args.features}")
            game_feat_pca = os.path.join(args.soccernet_dirpath, game, f"{half}_{args.features_PCA}")

            if not os.path.exists(game_feat_pca) or args.overwrite:
                feat = np.load(game_feat)
                feat = feat - average
                feat_reduced = pca.transform(feat)
                np.save(game_feat_pca, feat_reduced)
            else:
                print(f"{game_feat_pca} already exists")


    for game in tqdm(getListGames(["challenge"])):
        for half in [1,2]:
            game_feat = os.path.join(args.soccernet_dirpath, game, f"{half}_{args.features}")
            game_feat_pca = os.path.join(args.soccernet_dirpath, game, f"{half}_{args.features_PCA}")
            if not os.path.exists(game_feat_pca) or args.overwrite:
                feat = np.load(game_feat)
                
                feat = feat - average
                
                feat_reduced = pca.transform(feat)

                np.save(game_feat_pca, feat_reduced)

            else:
                print(f"{game_feat_pca} already exists")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Extract ResNet feature out of SoccerNet Videos.')

    parser.add_argument('--soccernet_dirpath', type=str, default="/media/giancos/Football/SoccerNet/",
                        help="Path for SoccerNet directory [default:/media/giancos/Football/SoccerNet/]")
    parser.add_argument('--features', type=str, default="ResNET_TF2.npy",
                        help="features to perform PCA on [default:ResNET_TF2.npy]")    
    parser.add_argument('--features_PCA', type=str, default="ResNET_TF2_PCA512.npy",
                        help="name of reduced features [default:ResNET_TF2_PCA512.npy]")
    parser.add_argument('--pca_file', type=str, default="pca_512_TF2.pkl",
                        help="pickle for PCA [default:pca_512_TF2.pkl]")
    parser.add_argument('--scaler_file', type=str, default="average_512_TF2.pkl",
                        help="pickle for average [default:average_512_TF2.pkl]")
    parser.add_argument('--dim_reduction', type=int, default=512,
                        help="dimension reduction [default:512]")

    parser.add_argument('--overwrite', action="store_true",
                        help="Overwrite the features? [default:False]")

    args = parser.parse_args()
    print(args)

    main(args)
