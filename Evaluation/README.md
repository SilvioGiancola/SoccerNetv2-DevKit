# Evaluation

We provide evaluation functions directly integrated in our pip package (`pip install SoccerNet`) as well as an evaluation server on [EvalAI](https://eval.ai/web/challenges/challenge-page/761/overview).

## Ouput Format

To submit your results on EvalAI or to use the integreted function of the pip package, the predictions of the network have to be saved in a specific format, with a specific folder structure.

```
Results.zip
 - league
   - season
     - game full name
       - results_spotting.json
       - results_segmentation.json
       - results_grounding.json
```

### Task 1: `results_spotting.json`

For the action spotting task, each json file needs to be constructed as follows:

```json
{
    "UrlLocal": "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal",
    "predictions": [ # list of predictions
        {
            "gameTime": "1 - 0:31", # format: "{half} - {minutes}:{seconds}",
            "label": "Ball out of play", # label for the spotting,
            "position": "31500", # time in milliseconds,
            "half": "1", # half of the game
            "confidence": "0.006630070507526398", # confidence score for the spotting,
        },
        {
            "gameTime": "1 - 0:39",
            "label": "Foul",
            "position": "39500",
            "half": "1",
            "confidence": "0.07358131557703018"
        },
        {
            "gameTime": "1 - 0:55",
            "label": "Foul",
            "position": "55500",
            "half": "1",
            "confidence": "0.20939764380455017"
        },
        ...
    ]
}
```

### Task 2: `results_segmentation.json`

TBD

### Task 3: `results_grounding.json`

For the replay grounding task, each json file needs to be constructed as follows:

```json
{
    "UrlLocal:": "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal", 
    "half1_time": 2700,# length of half1  in seconds
    "half2_time": 3043,# length of half1  in seconds
    "Replays": [ # list of replays
        {
            "half": 1, # half time
            "start": 346,  # start time of replay in seconds
            "end": 357,  # end time of replay  in seconds
            "detection": [ # list of predictions
                {
                    "time": 81.0, #time of detection in second
                    "score": 0.004524833057075739 # # confidence score 
                },
                {
                    "time": 164.5,
                    "score": 0.008931178599596024
                },
                {
                    "time": 246.0,
                    "score": 0.009728623554110527
                },

                ...
                ]
        },
        {
            "half": 1,
            "start": 532,
            "end": 541,
            "detection": [
                {
                    "time": 79.5,
                    "score": 0.008331561461091042
                },
                {
                    "time": 162.0,
                    "score": 0.02246086485683918
                },
                {
                    "time": 243.0,
                    "score": 0.02563951350748539
                },

                ...
                ]

        },
        ...      
        {
            "half": 2,
            "start": 2560,
            "end": 2566,
            "detection": [
                {
                    "time": 74.0,
                    "score": 0.02719578705728054
                },
                {
                    "time": 161.5,
                    "score": 0.023679519072175026
                },
                {
                    "time": 237.0,
                    "score": 0.05776015296578407
                },

                ...
                ]
        
        }

    ]
}
```
## How to evaluate locally the performances on the testing set

### Task 1: Spotting

```bash
python EvaluateSpotting.py --SoccerNet_path /path/to/SoccerNet/ --Predictions_path /path/to/SoccerNet/outputs/
```

```python
from SoccerNet.Evaluation.ActionSpotting import evaluate
results = evaluate(SoccerNet_path=PATH_DATASET, Predictions_path=PATH_PREDICTIONS,
                   split="test", version=2, prediction_file="results_spotting.json")

print("Average mAP: ", results["a_mAP"])
print("Average mAP per class: ", results["a_mAP_per_class"])
print("Average mAP visible: ", results["a_mAP_visible"])
print("Average mAP visible per class: ", results["a_mAP_per_class_visible"])
print("Average mAP unshown: ", results["a_mAP_unshown"])
print("Average mAP unshown per class: ", results["a_mAP_per_class_unshown"])
```

### Task 2: Segmentation

TBD

### Task 3: Grounding

```bash
python EvaluateGrounding.py --SoccerNet_path /path/to/SoccerNet/ --Predictions_path /path/to/SoccerNet/outputs/
```

```python
from SoccerNet.Evaluation.ReplayGrounding import evaluate

results = evaluate(SoccerNet_path=PATH_DATASET, Predictions_path=PATH_PREDICTIONS,
                   split="test", version=2, prediction_file="results_grounding.json")

print("Average mAP: ", results["a_mAP"])
print("Average mAP per class: ", results["a_mAP_per_class"])
print("Average mAP visible: ", results["a_mAP_visible"])
print("Average mAP visible per class: ", results["a_mAP_per_class_visible"])
print("Average mAP unshown: ", results["a_mAP_unshown"])
print("Average mAP unshown per class: ", results["a_mAP_per_class_unshown"])
```

## How to evaluate online the performances on the challenge

### Zip the results

```bash
cd /path/to/soccernet/outputs/
zip results_spotting.zip */*/*/results_spotting.json
zip results_segmentation.zip */*/*/results_segmentation.json
zip results_grounding.zip */*/*/results_grounding.json
```

### Visit [EvalAI](https://eval.ai/auth/login) to submit you zipped results
