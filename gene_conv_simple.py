'''
Baseline Model for ATAC-seq to CHIP-seq model
'''
# gflag for python
from absl import flags
# default packages
import copy
import time
import sys
import os
import random
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import quantile_transform
import numpy as np
# package for genomic data
from pybedtools import Interval, BedTool
from genomelake.extractors import ArrayExtractor, BigwigExtractor
# package for plotting
import matplotlib.pyplot as plt
# Keras
from keras.models import Sequential
from keras.layers import Conv1D, Dropout
from keras.callbacks import Callback
from keras import optimizers
from keras import backend as K
# custom utility package
from utils.compute_util import *
# custom file path package
from data import Data_Directories


# get flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('all', False, 'Process all intervals?')
flags.DEFINE_integer('sample_num', 1000, 'Number of samples', lower_bound=1)
flags.DEFINE_integer('window_size', 10001, 'Window size', lower_bound=2000)
flags.DEFINE_integer('save_freq', 100, 'Frequency for saving models/plots', lower_bound=1)
flags.DEFINE_integer('batch_size', 512, 'Batch size for training step')
flags.DEFINE_string('input_norm_scheme', 'coarse', 'Normalization scheme for inputs')
flags.DEFINE_string('output_norm_scheme', 'dl', 'Normalization scheme for outputs')
flags.DEFINE_string('loss', 'mse', 'Type of loss function to use')
flags.DEFINE_string('output_dir', 'Plots', 'Directory to save the plots')
flags.DEFINE_string('model_dir', 'Models', 'Directory to save the models')
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs for training')
FLAGS(sys.argv)

# default sample numbers, loss_type, and normalization scheme
sample_num = FLAGS.sample_num
loss_type = FLAGS.loss
input_norm_scheme = FLAGS.input_norm_scheme
output_norm_scheme = FLAGS.output_norm_scheme
window_size = FLAGS.window_size
save_freq = FLAGS.save_freq
batch_size = FLAGS.batch_size
process_all = FLAGS.all
output_dir = FLAGS.output_dir + "_" + output_norm_scheme
num_epochs = FLAGS.num_epochs
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "Pearson"))
model_dir = FLAGS.model_dir + "_" + output_norm_scheme
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
print "=" * 80
print "Configuration Details"
print "Type of Loss: {}, Input Normalization Scheme: {}".format(loss_type, input_norm_scheme)
print "Output Normalization Scheme: {}, Batch size: {}, Epochs: {}".format(output_norm_scheme, batch_size, num_epochs)
if not process_all:
    print "# of Samples: {}".format(sample_num)
print "=" * 80

# retrieve data
data = Data_Directories()

# get intervals for day0 data
day0_intervals = list(BedTool(data.intervals['day0']))
print '# of Intervals Extracted for day0: {}'.format(len(day0_intervals))

# get intervals for day3 data
day3_intervals = list(BedTool(data.intervals['day3']))
print '# of Intervals Extracted for day3: {}'.format(len(day3_intervals))

# get atac-seq data for day0 with 140 base pairs
bw_140bp_day0 = ArrayExtractor(data.input_atac['day0']['140'])
print 'Finished extracting bigwig for day0, 140bp'

# get atac-seq data for day0 with 140 base pairs
bw_140bp_day3 = ArrayExtractor(data.input_atac['day3']['140'])
print 'Finished extracting bigwig for day3, 140bp'

# normalize day0 intervals
normalized_day0_intervals = [normalize_interval(interval, window_size) for interval in day0_intervals if normalize_interval(interval, window_size)]
print 'Finished normalizing day0 intervals!'

# normalize day3 intervals
normalized_day3_intervals = [normalize_interval(interval, window_size) for interval in day3_intervals if normalize_interval(interval, window_size)]
print 'Finished normalizing day3 intervals!'
# TODO: temporary checking for invalid intervals
for u in normalized_day3_intervals[:]:
    try:
        bw_140bp_day0([u])
    except:
        normalized_day3_intervals.remove(u)
        pass

# fetch input (day0, ATAC-seq)
t0 = time.time()
# raw input shape: (number of samples, window_size, 5)
# coarse normalized input: (number of samples, window_size, 1)
inputs = None
if not process_all:
    normalized_day0_intervals = random.sample(normalized_day0_intervals, sample_num)
inputs = coarse_normalize_input(bw_140bp_day0(normalized_day0_intervals))
print inputs.shape
t1 = time.time()
print 'Time spent for getting signals of intervals for day0 atac-seq: {}'.format(t1-t0)

#fetch validation inputs (day3, ATAC-seq)
t2 = time.time()
val_inputs = None
if not process_all:
    normalized_day3_intervals = random.sample(normalized_day3_intervals, sample_num)
val_inputs = coarse_normalize_input(bw_140bp_day3(normalized_day3_intervals))
print val_inputs.shape
t3 = time.time()
print 'Time spent for getting signals of intervals for day3 atac-seq: {}'.format(t3-t2)

# fetch outputs (day0, histone)
histone_mark = BigwigExtractor(data.output_histone['day0']['H3K27ac'])
outputs = None
outputs = histone_mark(normalized_day0_intervals)
outputs = np.nan_to_num(outputs)
if output_norm_scheme == 'dl':
    outputs = double_log_transform(outputs)
elif output_norm_scheme == 'quant':
    outputs = quantile_transform(outputs, n_quantiles=50, random_state=7)
outputs = np.expand_dims(outputs, axis=2)
print 'Output Shape (of one sample): ', outputs[0].shape
print 'Expanded Output Shape: ', outputs[0].shape

# fetch validation outputs (day3, histone)
val_histone_mark = BigwigExtractor(data.output_histone['day3']['H3K27ac'])
val_outputs = None
val_outputs = val_histone_mark(normalized_day3_intervals)
val_outputs = np.nan_to_num(val_outputs)
if output_norm_scheme == 'dl':
    val_outputs = double_log_transform(val_outputs)
elif output_norm_scheme == 'quant':
    val_outputs = quantile_transform(val_outputs, n_quantiles=50, random_state=7)
val_outputs = np.expand_dims(val_outputs, axis=2)
print 'Output Shape (of one sample): ', val_outputs[0].shape
print 'Expanded Output Shape: ', val_outputs[0].shape

'''
CNN with 6 HIDDEN LAYERS and 1 OUTPUT LAYER
'''
# build convolutional network with keras
print "Building Keras sequential model..."
model = Sequential()
# 1) build hidden layer with 30 filters of size 5000
print "Adding the first hidden layer..."
hidden_filters_1 = 20
hidden_kernel_size_1 = 5000

model.add(Conv1D(
    filters=hidden_filters_1,
    kernel_size=hidden_kernel_size_1,
    padding='same',
    activation='relu',
    strides=1,
    input_shape=(window_size, 1)))

model.add(Dropout(0.1))

# 2) build hidden layer with 15 filters of size 300
print "Adding the second hidden layer..."
hidden_filters_2 = 15
hidden_kernel_size_2 = 2000

model.add(Conv1D(
    filters=hidden_filters_2,
    kernel_size=hidden_kernel_size_2,
    padding='same',
    activation='relu',
    strides=1))

model.add(Dropout(0.1))

# 3) building hidden layer with 5 filters of size 200
print "Adding the third hidden layer..."
hidden_filters_3 = 10
hidden_kernel_size_3 = 1000

model.add(Conv1D(
    filters=hidden_filters_3,
    kernel_size=hidden_kernel_size_3,
    padding='same',
    activation='relu',
    strides=1))

model.add(Dropout(0.1))

# 4) building hidden layer with 5 filters of size 200
print "Adding the third hidden layer..."
hidden_filters_4 = 5
hidden_kernel_size_4 = 200

model.add(Conv1D(
    filters=hidden_filters_4,
    kernel_size=hidden_kernel_size_4,
    padding='same',
    activation='relu',
    strides=1))

model.add(Dropout(0.1))

# 4) building hidden layer with 5 filters of size 200
print "Adding the third hidden layer..."
hidden_filters_5 = 3
hidden_kernel_size_5 = 50

model.add(Conv1D(
    filters=hidden_filters_5,
    kernel_size=hidden_kernel_size_5,
    padding='same',
    activation='relu',
    strides=1))

# 5) build output layer with 1 filter of size 20
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


print "Compiling a model with adam optimizer"
model.compile(loss=loss, optimizer=adam)

print model.summary()

# required for matplotlib
plt.switch_backend('agg')

# callback function for plotting loss graph for every 500 epochs
class Plot_Train_Loss_Callback(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.epochs = 0

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs['loss'])
        self.epochs += 1
        if self.epochs % save_freq == 0:
            # summarize history for loss
            plt.plot(range(self.epochs), self.losses)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig(os.path.join(output_dir, 'model_train_'+loss_type+"_"+str(self.epochs)+'.png'))
            self.model.save(os.path.join(model_dir, 'model_'+loss_type+"_"+str(self.epochs)+".h5"))
            plt.close()

class Compute_Pearson_Callback(Callback):
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val

    def on_train_begin(self, logs={}):
        self.p_train_history, self.p_val_history = [], []
        self.epochs = 0

    def on_epoch_end(self, batch, logs={}):
        self.epochs += 1
        x_train, y_train = self.x_train, self.y_train
        x_val, y_val = self.x_val, self.y_val
        y_pred_train = self.model.predict(x_train)
        p_train_list = []
        for y_t, y_pt in zip(y_train, y_pred_train):
            p_train_list.append(pearsonr(y_t.squeeze(), y_pt.squeeze()))
        p_train = np.mean(p_train_list)
        y_pred_val = self.model.predict(x_val)
        p_val_list = []
        for y_v, y_pv in zip(y_val, y_pred_val):
            p_val_list.append(pearsonr(y_v.squeeze(), y_pv.squeeze()))
        p_val = np.mean(p_val_list)
        print "Train Pearson Corr: {}, Valid Pearson Corr: {}".format(p_train, p_val)
        self.p_train_history.append(p_train)
        self.p_val_history.append(p_val)
        if self.epochs % save_freq == 0:
            # record pearson correlation for train
            plt.plot(range(self.epochs), self.p_train_history)
            plt.title('Pearson Correlation - Train')
            plt.ylabel('pearson correlation')
            plt.xlabel('epoch')
            plt.savefig(os.path.join(output_dir, 'Pearson', 'train_'+loss_type+"_"+str(self.epochs)+'.png'))
            plt.close()
            plt.plot(range(self.epochs), self.p_val_history)
            plt.title('Pearson Correlation - Val')
            plt.ylabel('pearson correlation')
            plt.xlabel('epoch')
            plt.savefig(os.path.join(output_dir, 'Pearson', 'val_'+loss_type+"_"+str(self.epochs)+'.png'))
            plt.close()


print "Fitting the model..."
model.fit(inputs, outputs, batch_size=batch_size, epochs=num_epochs, callbacks=[Plot_Train_Loss_Callback(), Compute_Pearson_Callback(inputs, outputs, val_inputs, val_outputs)], validation_data=(val_inputs, val_outputs), shuffle=True)

