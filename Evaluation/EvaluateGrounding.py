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
