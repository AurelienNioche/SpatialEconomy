import argparse

parser = argparse.ArgumentParser()
parser.add_argument('parameters', type=str,
                    help='A name of pickle file for paramaters is required!')

args = parser.parse_args()
print(args.parameters)