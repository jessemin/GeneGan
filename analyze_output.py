'''
Analysis script for output value distribution
'''
# gflag for python
from absl import flags
# default packages
import copy
import time
import sys
import numpy as np
# package for genomic data
from pybedtools import Interval, BedTool
from genomelake.extractors import ArrayExtractor, BigwigExtractor
from data import Data_Directories
# pacakge for plotting
import matplotlib.pyplot as plt
# custom utility package
from utils.compute_util import *
from sklearn.preprocessing import quantile_transform


# get flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('sample_num', 267226, 'Number of samples', lower_bound=1)
flags.DEFINE_string('bin', 'doane', 'Type of histogram to use')
FLAGS(sys.argv)

# default sample numbers
sample_num = FLAGS.sample_num
bin_type = FLAGS.bin

# normalizes intervals to 1000-bp bins with summit at center
def normalize_interval(interval):
    normalized_interval = copy.deepcopy(interval)
    summit = int(interval.start) + int(interval[-1])
    normalized_interval.start = summit-1000
    normalized_interval.end = summit+1000
    return normalized_interval


# retrieve data
data = Data_Directories()

# get intervals for day0 data
intervals = list(BedTool(data.intervals['day0']))
print '# of Intervals: {}'.format(len(intervals))

# normalize intervals
normalized_intervals = [normalize_interval(interval) for interval in intervals]
print 'Finished normalizing intervals: {}'.format(len(normalized_intervals))

# fetch entire outputs
histone_mark = BigwigExtractor(data.output_histone['day0']['H3K27ac'])
outputs = histone_mark(normalized_intervals[:sample_num])
print 'Output Shape: ', outputs[0].shape
outputs = np.nan_to_num(outputs.reshape((-1)))
dl_outputs = double_log_transform(outputs)
q_outputs = quantile_transform(outputs.reshape(-1, 1), n_quantiles=10, random_state=0)
print "quantile transform shape: {}".format(q_outputs.shape)
print 'Flattened Shape: ', outputs.shape

print "Original"
hist, bins = np.histogram(outputs, bin_type)
print zip(bins, hist)
print "Double Log Transform"
hist2, bins2 = np.histogram(dl_outputs, bins)
print zip(bins2, hist2)
print "Re-bin double log transform"
hist3, bins3 = np.histogram(dl_outputs, bin_type)
print zip(bins3, hist3)
print "Quantile transform"
hist4, bins4 = np.histogram(q_outputs, bin_type)
print zip(bins4, hist4)

print "Generating histogram..."
plt.switch_backend('agg')
plt.bar(bins[:-1], hist)

#plt.bar(new_bins[:-1], new_hist)
plt.savefig('analyses/output_analysis_'+str(sample_num)+'.png')
plt.close()
