from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from SoccerNet.Evaluation.ActionSpotting import evaluate

if __name__ == '__main__':

    # Load the arguments
    parser = ArgumentParser(description='Evaluation for Action Spotting', 
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--SoccerNet_path', required=True,
                        type=str, help='Path to the SoccerNet-V2 dataset folder (or zipped file) with labels')
    parser.add_argument('--Predictions_path', required=True,
                        type=str, help='Path to the predictions folder (or zipped file) with prediction')
    parser.add_argument('--Prediction_file', required=False, type=str,
                        help='Name of the prediction files as stored in folder (or zipped file) [None=try to infer it]', default=None)
    parser.add_argument('--split', required=False, type=str,
                        help='Set on which to evaluate the performances', default="test")
    parser.add_argument('--framerate', required=False, type=int,
                        help='Framerate of the input features', default=2)

    args = parser.parse_args()

    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = \
        evaluate(SoccerNet_path=args.SoccerNet_path, Predictions_path=args.Predictions_path, split=args.split,
                 framerate=args.framerate, prediction_file=args.Prediction_file)

    print("Average mAP: ", a_mAP)
    print("Average mAP visible: ", a_mAP_visible)
    print("Average mAP unshown: ", a_mAP_unshown)
    print("Average mAP per class: ", a_mAP_per_class)
    print("Average mAP visible per class: ", a_mAP_per_class_visible)
    print("Average mAP unshown per class: ", a_mAP_per_class_unshown)

# python tools/EvaluateSpotting.py --Predictions_path EvalAI/submission/results_spotting/ --SoccerNet_path /media/giancos/Football/SoccerNet/ --Prediction_file Predictions-v2.json
# python tools/EvaluateSpotting.py --Predictions_path EvalAI/submission/results_spotting.zip --SoccerNet_path EvalAI/annotations/test_annotations_spotting.zip --Prediction_file Predictions-v2.json
