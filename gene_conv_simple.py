import copy
import time
import sys
import os
import numpy as np
from pybedtools import Interval
from pybedtools import BedTool
from genomelake.extractors import ArrayExtractor, BigwigExtractor
from data import Data_Directories
import matplotlib.pyplot as plt
# import Keras
from keras.models import Sequential
from keras.layers import Conv1D, Dropout
from keras.callbacks import Callback
from keras import optimizers
from keras import backend as K


# normalizes intervals to 2000-bp bins with summit at center
def normalize_interval(interval):
    normalized_interval = copy.deepcopy(interval)
    summit = int(interval.start) + int(interval[-1])
    normalized_interval.start = summit-1000
    normalized_interval.end = summit+1000
    return normalized_interval


# function that transforms output target 
def double_log_transform(d):
    return np.log(1.0+np.log(1.0+d))


# TODO: Need to check
# https://chat.stackoverflow.com/rooms/156491/discussion-between-julio-daniel-reyes-and-eleanora
# https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
def pearson_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(np.multiply(xm,ym))
    r_den = K.sqrt(np.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


number = 267226
loss_type = 'pearson'
if len(sys.argv) != 1 and len(sys.argv) != 3:
    print "Error: Wrong number of parameters!"
    print "  1) python gene_simple_conv.py"
    print "  2) python gene_simple_conv.py (number of samples) (type of loss)"
    print "     ex) python gene_simple_conv.py 5000 'mse'"
    sys.exit(-1)
if len(sys.argv) == 3:
    number = int(sys.argv[1])
    loss_type = sys.argv[2]
print "Configuration"
print "Number of Samples: {}, Type of Loss: {}".format(number, loss_type)
print "="*60

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

# fetch inputs
t0 = time.time()
# shape: (number of samples, 2000, 5)
inputs = bw_140bp_day0(normalized_intervals[:number])
print inputs[0]
t1 = time.time()
print 'Time spent for getting signals of intervals: {}'.format(t1-t0)

# fetch outputs
histone_mark = BigwigExtractor(data.output_histone['day0']['H3K27ac'])
outputs = histone_mark(normalized_intervals[:number])
outputs = np.nan_to_num(outputs)
outputs = double_log_transform(outputs)
print 'Output Shape: ', outputs[0].shape
outputs = np.expand_dims(outputs, axis=2)
print 'Expanded Output Shape: ', outputs[0].shape

'''
CNN with 3 HIDDEN LAYERS and 1 OUTPUT LAYER
'''
# build convolutional network with keras
print "Building Keras sequential model..."
model = Sequential()
# 1) build hidden layer with 10 filters of size 500
print "Adding the first hidden layer..."
hidden_filters_1 = 15
hidden_kernel_size_1 = 600

model.add(Conv1D(
    filters=hidden_filters_1,
    kernel_size=hidden_kernel_size_1,
    padding='same',
    activation='relu',
    strides=1,
    input_shape=(2000, 5)))

model.add(Dropout(0.2))

# 2) build hidden layer with 7 filters of size 300
print "Adding the second hidden layer..."
hidden_filters_2 = 7
hidden_kernel_size_2 = 300

model.add(Conv1D(
    filters=hidden_filters_2,
    kernel_size=hidden_kernel_size_2,
    padding='same',
    activation='relu',
    strides=1))

model.add(Dropout(0.2))

# 3) building hidden layer with 5 filters of size 200
print "Adding the third hidden layer..."
hidden_filters_3 = 5
hidden_kernel_size_3 = 100

model.add(Conv1D(
    filters=hidden_filters_3,
    kernel_size=hidden_kernel_size_3,
    padding='same',
    activation='relu',
    strides=1))

model.add(Dropout(0.1))

# 4) build output layer with 1 filter of size 20
# NOTE: linear activation for the final layer
print "Adding a output layer..."
output_filters = 1
output_kernel_size = 20
model.add(Conv1D(filters=output_filters,
    kernel_size=output_kernel_size,
    padding='same',
    activation='linear',
    strides=1
    ))


# setting optimizer
adam = optimizers.Adam(clipnorm=1.)

# setting loss type
loss = pearson_loss
if loss_type == 'pearson':
    loss = pearson_loss
if loss_type == 'mse':
    loss = 'mean_squared_error'


print "Compiling a model with adam optimizer with MSE..."
model.compile(loss=loss, optimizer=adam)

print model.summary()

# required for matplotlib
plt.switch_backend('agg')

# callback function for plotting loss graph for every 500 epochs
class Loss_Plot_Callback(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.epochs = 0

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs['loss'])
        self.epochs += 1
        if self.epochs % 10 == 0:
            # summarize history for loss
            plt.plot(range(self.epochs), self.losses)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig(os.path.join('Plots', 'model_loss_simple_conv_'+loss_type+"_"+str(self.epochs)+'.png'))
            self.model.save(os.path.join('Models', 'model_'+loss_type+"_"+str(self.epochs)+".h5"))

num_epochs = 1000000
print "Fitting the model..."
model.fit(inputs, outputs, batch_size=512, epochs=num_epochs, callbacks=[Loss_Plot_Callback()])
