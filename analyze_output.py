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
outputs = outputs.reshape((-1))
print 'Flattened Shape: ', outputs.shape

print np.argwhere(np.isnan(outputs))
print "filtering NaN..."
outputs = filter(lambda v: v==v, outputs)
print np.argwhere(np.isnan(outputs))
print "filtering done!"
hist, bins = np.histogram(outputs, bin_type)
print zpi(bins, hist)
#print "writing before hist"
print "Making histogram..."
plt.switch_backend('agg')
plt.bar(bins[:-1], hist)
'''
plt.savefig('analyses/test_before_analysis_'+str(sample_num)+'.png')
plt.clf()

new_hist, new_bins = [], []
first = True
for i, bin in enumerate(bins):
    if i == len(bins)-1:
        continue
    if bin <= 20:
        new_hist.append(hist[i])
        new_bins.append(bin)
    if bin > 20:
        if first:
            new_bins.append(bin)
            new_bins.append(bins[i+1])
            new_hist.append(hist[i])
            first = False
        else:
            new_hist[-1] += hist[i]
print '[After] Number of bins: ', len(new_bins)
print '[After] Number of hist: ', len(new_hist)
print new_bins
'''
#plt.bar(new_bins[:-1], new_hist)
plt.savefig('analyses/output_analysis_'+str(sample_num)+'.png')
plt.close()
