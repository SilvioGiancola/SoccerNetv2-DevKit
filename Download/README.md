# How to download SoccerNet?

You can use the pip package for that purpose:

```bash
pip install SoccerNet --upgrade
```

Then usethe API to downlaod the data of interest:

```
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="/path/to/SoccerNet")

mySoccerNetDownloader.password = input("Password for videos?:\n")

mySoccerNetDownloader.downloadGames(files=["Labels-v2.json", "1_ResNET_TF2.npy", "2_ResNET_TF2.npy"],
                                    split=["train", "valid", "test", "challenge"])  # download Features and Spotting Annotations
```

## What data are availabel for Download?

### Annotations

- **Labels-v2.json**: Labels from SoccerNet-v2 - action spotting
- **Labels-camera.json**: Labels from SoccerNet-v1 - camera shot segmentation
- **Labels.json**: Labels from SoccerNet-v1 - action spotting for goals/cards/subs only

## Videos (requires a password to download - please fill this [NDA](https://soccer-net.org) to request access)

- **1_HQ.mkv**: HQ video 1st half
- **2_HQ.mkv**: HQ video 2nd half
- **video.ini**: information on start/duration for each half of the game in the HQ video, in second
- **1.mkv**: LQ video 1st half - timmed with start/duration from HQ video - resolution 224*398 - 25 fps
- **2.mkv**: LQ video 2nd half - timmed with start/duration from HQ video - resolution 224*398 - 25 fps

## Pre-extracted Features

- **1_ResNET_TF2.npy**: ResNET features @2fps for 1st half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Features)
- **2_ResNET_TF2.npy**: ResNET features @2fps for 2nd half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Features)
- **1_ResNET_TF2_PCA512.npy**: ResNET features @2fps for 1st half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Features), with dimensionality [reduced to 512 using PCA](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/main/Features/ReduceFeaturesPCA.py)
- **2_ResNET_TF2_PCA512.npy**: ResNET features @2fps for 2nd half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Features), with dimensionality [reduced to 512 using PCA](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/main/Features/ReduceFeaturesPCA.py)
- **1_ResNET_5fps_TF2.npy**: ResNET features @5fps for 1st half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Features)
- **2_ResNET_5fps_TF2.npy**: ResNET features @5fps for 2nd half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Features)
- **1_ResNET_5fps_TF2_PCA512.npy**: ResNET features @5fps for 1st half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Features), with dimensionality [reduced to 512 using PCA](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/main/Features/ReduceFeaturesPCA.py)
- **2_ResNET_5fps_TF2_PCA512.npy**: ResNET features @5fps for 2nd half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Features), with dimensionality [reduced to 512 using PCA](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/main/Features/ReduceFeaturesPCA.py)
- **1_ResNET_25fps_TF2.npy**: ResNET features @25fps for 1st half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Features)
- **2_ResNET_25fps_TF2.npy**: ResNET features @25fps for 2nd half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Features)

### Legacy from SoccerNet-v1

- **1_C3D.npy**: C3D features @2fps for 1st half from SoccerNet-v1
- **2_C3D.npy**: C3D features @2fps for 2nd half from SoccerNet-v1
- **1_C3D_PCA512.npy**: C3D features @2fps for 1st half from SoccerNet-v1, with dimensionality reduced to 512 using PCA
- **2_C3D_PCA512.npy**: C3D features @2fps for 2nd half from SoccerNet-v1, with dimensionality reduced to 512 using PCA
- **1_I3D.npy**: I3D features @2fps for 1st half from SoccerNet-v1
- **2_I3D.npy**: I3D features @2fps for 2nd half from SoccerNet-v1
- **1_I3D_PCA512.npy**: I3D features @2fps for 1st half from SoccerNet-v1, with dimensionality reduced to 512 using PCA
- **2_I3D_PCA512.npy**: I3D features @2fps for 2nd half from SoccerNet-v1, with dimensionality reduced to 512 using PCA
- **1_ResNET.npy**: ResNET features @2fps for 1st half from SoccerNet-v1
- **2_ResNET.npy**: ResNET features @2fps for 2nd half from SoccerNet-v1
- **1_ResNET_PCA512.npy**: ResNET features @2fps for 1st half from SoccerNet-v1, with dimensionality reduced to 512 using PCA
- **2_ResNET_PCA512.npy**: ResNET features @2fps for 2nd half from SoccerNet-v1, with dimensionality reduced to 512 using PCA