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

number = 267226
histone_mark = BigwigExtractor(data.output_histone['day0']['H3K27ac'])
outputs = histone_mark(normalized_intervals[:number])
print 'Output Shape: ', outputs[0].shape


def double_log_transform(data):
    return np.log(1.0+np.log(1.0+data))


print "Testing double log transformation..."
print "1 -> ", double_log_transform(1)
print "5000 -> ", double_log_transform(5000)
print "=" * 40
print "Transforming actual histone mark..."
start = time.time()
transformed_outputs = double_log_transform(outputs)
end = time.time()
print "Time spent for transforming ", number," samples: ", end-start
print transformed_outputs.shape
print transformed_outputs
