import copy
import time
import numpy as np
from pybedtools import Interval
from pybedtools import BedTool
from genomelake.extractors import ArrayExtractor, BigwigExtractor
from data import Data_Directories
import matplotlib.pyplot as plt
# import Keras
from keras.models import Sequential
from keras.layers import Conv1D
from keras.callbacks import Callback
from keras import optimizers


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

# get atac-seq data for day0 with 140 base pairs
bw_140bp_day0 = ArrayExtractor(data.input_atac['day0']['140'])
print 'Finished extracting bigwig for day0, 140bp'

# normalize intervals
normalized_intervals = [normalize_interval(interval) for interval in intervals]
print 'Finished normalizing intervals: {}'.format(len(normalized_intervals))
print 'First ten normalized examples...'
print normalized_intervals[:10]

# fetch 50000 inputs
t0 = time.time()
# shape: (number of samples, 2000, 5)
inputs = bw_140bp_day0(normalized_intervals[:5000])
t1 = time.time()
print 'Time spent for getting signals of intervals: {}'.format(t1-t0)

# fetch 50000 outputs
histone_mark = BigwigExtractor(data.output_histone['day0']['H3K27ac'])
outputs = histone_mark(normalized_intervals[:5000])
print 'Output Shape: ', outputs[0].shape
outputs = np.expand_dims(outputs, axis=2)
print 'Expanded Output Shape: ', outputs[0].shape

'''
SIMPLE CONV NET with 1 HIDDEN LAYER
'''
# build convolutional network with keras
print "Building Keras sequential model..."
model = Sequential()
# 1) build hidden layer with 15 filters of size 200
print "Adding a hidden layer..."
hidden_filters = 15
hidden_kernel_size = 200

model.add(Conv1D(
    filters=hidden_filters,
    kernel_size=hidden_kernel_size,
    padding='same',
    activation='relu',
    strides=1,
    input_shape=(2000, 5)))

# 2) build output layer with 1 filter of size 20
print "Adding a output layer..."
output_filters = 1
output_kernel_size = 20
model.add(Conv1D(filters=output_filters,
    kernel_size=output_kernel_size,
    padding='same',
    activation='relu',
    strides=1
    ))

print "Compiling a model with adam optimizer with MSE..."
model.compile(loss='mean_squared_error', optimizer='adam')

print model.summary()

print "Fitting the model..."
history = model.fit(inputs, outputs, batch_size=128, epochs=20000)

plt.switch_backend('agg')

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('model_loss.png')
