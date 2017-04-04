import sys
import data.amuse as amuse

import argparse

parser = argparse.ArgumentParser(description='Pre-process amuse data')
parser.add_argument('--subject',type=str)
args = parser.parse_args()
data = amuse.preprocess_amuse_mat(args.subject)

