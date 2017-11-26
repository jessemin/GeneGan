import copy
import time
import numpy as np
from pybedtools import Interval
from pybedtools import BedTool
from genomelake.extractors import ArrayExtractor, BigwigExtractor
from data import Data_Directories
import matplotlib.pyplot as plt


# normalizes intervals to 2000-bp bins with summit at center
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
print 'First ten raw intervals...'
print intervals[:10]


# normalize intervals
normalized_intervals = [normalize_interval(interval) for interval in intervals]
print 'Finished normalizing intervals: {}'.format(len(normalized_intervals))
print 'First ten normalized examples...'
print normalized_intervals[:10]

number = 100000
# fetch 50000 outputs
histone_mark = BigwigExtractor(data.output_histone['day0']['H3K27ac'])
outputs = histone_mark(normalized_intervals[:number])
print 'Output Shape: ', outputs[0].shape
outputs = outputs.reshape((-1))
print 'Flattened Shape: ', outputs.shape

print np.argwhere(np.isnan(outputs))
print "filtering NaN..."
outputs = filter(lambda v: v==v, outputs)
print np.argwhere(np.isnan(outputs))
print "filtering done!"
hist, bins = np.histogram(outputs, 'auto')
plt.switch_backend('agg')
plt.bar(bins[:-1], hist)
plt.savefig('analyses/before_analysis_'+str(number)+'.png')
plt.clf()
print '[Before]  Number of bins: ', len(hist)
new_hist, new_bins = [], []
first = True
for i, bin in enumerate(bins):
    if i == len(bins)-1:
        continue
    if bin <= 100:
        new_hist.append(hist[i])
        new_bins.append(bin)
    if bin > 100:
        if first:
            new_bins.append(bin)
            new_bins.append(bins[i+1])
            new_hist.append(hist[i])
            first = False
        else:
            new_hist[-1] += hist[i]
print '[After] Number of bins: ', len(new_bins)
print '[After] Number of hist: ', len(new_hist)
plt.bar(new_bins[:-1], new_hist)
plt.savefig('analyses/analysis_'+str(number)+'.png')
plt.clf()
