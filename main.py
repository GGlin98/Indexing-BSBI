import argparse

from Indexer import *

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', type=int, default='512', help='Block size (an integer)')
parser.add_argument('-u', '--unit', type=str, default='k', choices='KMGkmg', help='Block size unit, in [K, M, G]')
parser.add_argument('-d', '--dir', type=str, default='HillaryEmails', help='The directory path for input documents')
parser.add_argument('-o', '--output', type=str, default='Output', help='The output directory path')
parser.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Track and display memory usage (will degrade the performance)')
args = parser.parse_args()
args.unit = args.unit.upper()

if args.unit == 'G':
    args.size *= (1024 ** 3)
elif args.unit == 'M':
    args.size *= (1024 ** 2)
elif args.unit == 'K':
    args.size *= 1024

args.output = 'Output'

indexer = IndexerBSBI(args.dir, args.size, args.output, args.verbose)
indexer.construct_index()
