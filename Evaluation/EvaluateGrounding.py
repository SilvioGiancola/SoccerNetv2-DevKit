from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from SoccerNet.Evaluation.ReplayGrounding import evaluate

if __name__ == '__main__':

    # Load the arguments
    parser = ArgumentParser(description='Evaluation for Replay Grounding',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--SoccerNet_path',   required=False, type=str,
                        default="/media/giancos/Football/SoccerNet/",     help='Path for SoccerNet')
    parser.add_argument('--Predictions_path',   required=False, type=str,
                        default="/media/giancos/Football/SoccerNet/",     help='Path for Output_results')
    parser.add_argument('--Prediction_file', required=False, type=str,
                        help='Name of the prediction files as stored in folder (or zipped file) [None=try to infer it]', default=None)
    parser.add_argument('--split',   required=False, type=str,
                        default="test", help='Set on which to evaluate the performances')

    args = parser.parse_args()

    results = evaluate(SoccerNet_path=args.SoccerNet_path, Predictions_path=args.Predictions_path,
                    split=args.split, prediction_file=args.Prediction_file)

    print("Average AP: ", results["a_AP"])
