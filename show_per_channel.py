from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import numpy as np
import subprocess
import bcolz
import json

from genomelake.backend import load_directory

# Setup logging
log_formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s] %(message)s')
logger = logging.getLogger('perchannelGenomelake')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(log_formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

_blosc_params = bcolz.cparams(clevel=5, shuffle=bcolz.SHUFFLE, cname='lz4')

_array_writer = {
            'numpy': lambda arr, path: np.save(path, arr),
            'bcolz': lambda arr, path: bcolz.carray(
                arr, rootdir=path, cparams=_blosc_params, mode='w').flush()
            }


def parse_args():
    parser = argparse.ArgumentParser(description='Convert 1D genomelake data directory to a bigwig.')

    parser.add_argument("--data_dir",
                        type=str,
                        required=False,
                        default='/srv/scratch/csfoo/projects/ggr/bcolz/atac/Day-0.0-merged.trim.PE2SE.fixed.nodup.namesorted.fragments.bed.gz-140bp-nuc',
                        help="Genomelake data directory.")
    parser.add_argument("--chrom_sizes",
                        type=str,
                        required=False,
                        default="mnt/data/annotations/by_release/hg19.GRCh37/hg19.chrom.sizes",
                        help="Chrommosome sizes.")
    parser.add_argument("--output_prefix",
                        type=str,
                        required=False,
                        default="/srv/scratch/jesikmin/out",
                        help="bigwig prefix. Example: a `outfile` prefix results in a `outfile.bw` bigwig file")

    args = parser.parse_args()
    return args


# parse args
args = parse_args()
print(args)
bigwig = '{}.bw'.format(args.output_prefix)

# load data directory
logger.info("Loading genomelake data..")
data = load_directory(args.data_dir, in_memory=True)

file_shapes = {}
for chrom, chrom_data in data.items():
    logger.info("Chrom " + str(chrom) + "...")
    for _channel_idx in range(5):
        channel = np.copy(chrom_data._arr[:,_channel_idx])
        output_path = os.path.join("/srv/scratch/jesikmin/temp/" + str(_channel_idx), chrom)
        os.makedirs(output_path)
        _array_writer['bcolz'](channel.astype(np.float32),
                               output_path)
        file_shapes[chrom] = (chrom_data._arr.shape[0], )
for idx in range(5):
    with open(os.path.join("/srv/scratch/jesikmin/temp/"+str(idx), 'metadata.json'), 'w') as fpp:
        json.dump({'file_shapes': file_shapes,
                   'type': 'array_{}'.format('bcolz'),
                   'source': bigwig}, fpp)

logger.info("Done!")
