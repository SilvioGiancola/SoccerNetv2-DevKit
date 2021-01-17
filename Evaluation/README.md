# Evaluation

We provide evaluation functions on our pip package (`pip install SoccerNet`) as well as an evaluation server on [EvalAI]().

## Ouput Format

```
Results.zip
 - league
   - season
     - game full name
       - results_spotting.json
       - results_segmentation.json
       - results_ground.json
```

### Task 1: `results_spotting.json`

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

TBD

## How to evaluate locally the performances on the testing set

### Task 1: Spotting

```bash
python tools/EvaluateSpotting.py --SoccerNet_path /path/to/SoccerNet/ --Predictions_path /path/to/SoccerNet/outputs/
```

```python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from SoccerNet.Evaluation.ActionSpotting import evaluate

if __name__ == '__main__':

    # Load the arguments
    parser = ArgumentParser(description='Evaluation for Action Spotting',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--SoccerNet_path',   required=True, type=str, help='Path to the SoccerNet-V2 dataset folder' )
    parser.add_argument('--Predictions_path',   required=True, type=str, help='Path to the predictions folder' )
    parser.add_argument('--split',   required=False, type=str, default= "test", help='Set on which to evaluate the performances' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )

    args = parser.parse_args()

    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = evaluate(args.SoccerNet_path, args.Predictions_path, args.split, args.framerate)

    print("Average mAP: ", a_mAP)
    print("Average mAP visible: ", a_mAP_visible)
    print("Average mAP unshown: ", a_mAP_unshown)
    print("Average mAP per class: ", a_mAP_per_class)
    print("Average mAP visible per class: ", a_mAP_per_class_visible)
    print("Average mAP unshown per class: ", a_mAP_per_class_unshown)
```

### Task 2: Segmentation

TBD

### Task 3: Grounding

```bash
python tools/EvaluateReplay.py --SoccerNet_path /path/to/SoccerNet/ --Predictions_path /path/to/SoccerNet/outputs/
```

```python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from SoccerNet.Evaluation.ReplayGrounding import evaluate

if __name__ == '__main__':

    # Load the arguments
    parser = ArgumentParser(description='Evaluation for Replay Grounding',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--SoccerNet_path',   required=False, type=str,   default="/media/giancos/Football/SoccerNet/",     help='Path for SoccerNet' )
    parser.add_argument('--Predictions_path',   required=False, type=str,   default="/media/giancos/Football/SoccerNet/",     help='Path for Output_results' )
    parser.add_argument('--split',   required=False, type=str, default= "test", help='Set on which to evaluate the performances' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )

    args = parser.parse_args()

    results = evaluate(args.SoccerNet_path, args.Predictions_path, args.split, args.framerate)

    print(results)
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
