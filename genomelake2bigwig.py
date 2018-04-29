from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import numpy as np
import subprocess

from genomelake.backend import load_directory

# Setup logging
log_formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s] %(message)s')
logger = logging.getLogger('genomelake2bigwig')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(log_formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


def parse_args():
    parser = argparse.ArgumentParser(description='Convert 1D genomelake data directory to a bigwig.')

    parser.add_argument("--data_dir",
                        type=str,
                        help="Genomelake data directory.")
    parser.add_argument("--chrom_sizes",
                        type=str,
                        default="/mnt/data/annotations/by_release/hg19.GRCh37/hg19.chrom.sizes",
                        help="Chrommosome sizes.")
    parser.add_argument("--output_prefix", type=str,
                        help="bigwig prefix. Example: a `outfile` prefix results in a `outfile.bw` bigwig file")

    args = parser.parse_args()
    return args


def repeating(data):
    """
    Partitions array into list of arrays with consecutive repeating numbers
    """
    return np.split(data, np.where(np.diff(data) != 0)[0]+1)


# parse args
args = parse_args()
bedgraph = '{}.bedgraph'.format(args.output_prefix)
bigwig = '{}.bw'.format(args.output_prefix)

# load data directory
logger.info("Loading genomelake data..")
data = load_directory(args.data_dir, in_memory=True)

with open(bedgraph, 'w') as fp:
    for chrom, chrom_data in data.items():
        start = 0
        end = 0
        repeating_chrom_data = repeating(chrom_data)
        for i, arr in enumerate(repeating_chrom_data):
            if i != 0:
                start = end
            end += arr.shape[0]
            val = arr[0]
            fp.write('{}\t{}\t{}\t{}\n'.format(chrom, start, end, val))
logger.info("Wrote bedgraph.")

try:
    output = subprocess.check_output(
        ['wigToBigWig', bedgraph, args.chrom_sizes, bigwig],
        stderr=subprocess.STDOUT)
    logger.info('wigToBigWig output: {}'.format(output))
except subprocess.CalledProcessError as e:
    logger.error('wigToBigWig terminated with exit code {}'.format(
        e.returncode))
    logger.error('output was:\n' + e.output)

logger.info('Wrote bigwig.')
