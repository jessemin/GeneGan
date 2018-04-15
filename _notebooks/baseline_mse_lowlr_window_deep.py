
# coding: utf-8

# # Baseline with mse loss with wide cnn modified + adam
# #### Convert histone mark signals and use deep CNN for training.

# #### NOTE: Need to activate genomelake environment before this code. Simply type 'genomelake' in terminal.

# In[1]:


#get_ipython().magic(u'env CUDA_VISIBLE_DEVICES=4,5')
import os, sys
sys.path.append("..")
import random
# custom file path package
from data import Data_Directories
# custom utility package
from utils.compute_util import *
# package for genomic data
from pybedtools import Interval, BedTool
from genomelake.extractors import ArrayExtractor, BigwigExtractor
# package for plotting
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
from scipy.stats.stats import pearsonr,spearmanr
import tensorflow as tf


# In[2]:


window_size = 5001
process_all = False
sample_num = 1000


# In[3]:


# retrieve data
data = Data_Directories()
print data.intervals.keys()
print data.input_atac['day0'].keys()
print data.output_histone['day0'].keys()


# In[4]:


# get intervals for day0 data
day0_intervals = list(BedTool(data.intervals['day0']))
print '# of Intervals Extracted for day0: {}'.format(len(day0_intervals))


# In[5]:


# create an ArrayExtractor for ATAC-seq for day0 with 140 base pairs
bw_140bp_day0 = ArrayExtractor(data.input_atac['day0']['140'])
print 'Finished extracting bigwig for day0, 140bp'


# In[6]:


# create a BigWigExtractor for histone makr 'H3K27ac' for day0
bw_histone_mark_day0 = BigwigExtractor(data.output_histone['day0']['H3K27ac'])
print 'Finished extracting bigwig for day0, 140bp'


# In[7]:


# normalize day0 intervals
normalized_day0_intervals = [normalize_interval(interval, window_size) for interval in day0_intervals if normalize_interval(interval, window_size)]
print 'Finished normalizing day0 intervals!'


# In[8]:


assert (len(day0_intervals)==len(normalized_day0_intervals))
print "Examples of original intervals"
print [(int(_interval.start)+int(_interval[-1]), [int(_interval.start), int(_interval.end)])
       for _interval in day0_intervals[:3]]
print "Examples of normalized intervals with window size of {}".format(window_size)
print [([int(_interval.start), int(_interval.end)])
       for _interval in  normalized_day0_intervals[:3]]


# In[9]:


atac_seq_day0 = bw_140bp_day0(normalized_day0_intervals)
print atac_seq_day0.shape


# In[10]:


#TODO: put this into utils if possible
def prune_invalid_intervals(intervals, bigwig_file):
    for _interval in intervals[:]:
        try:
            bigwig_file([_interval])
        except:
            intervals.remove(_interval)
            pass
        
print "Before pruning day0: {}".format(len(normalized_day0_intervals))
prune_invalid_intervals(normalized_day0_intervals, bw_140bp_day0)
print "After pruning day0: {}".format(len(normalized_day0_intervals))


# In[11]:


print "Dimension of ATAC-seq signal: {}".format(bw_140bp_day0(normalized_day0_intervals[:1]).shape)


# In[12]:


print "Dimension of histone mark signal: {}".format(bw_histone_mark_day0(normalized_day0_intervals[:1]).shape)


# In[13]:


# replace nan values with zeros and convert it to p-values
histone_mark_day0 = np.nan_to_num(bw_histone_mark_day0(normalized_day0_intervals))
print histone_mark_day0.shape


# In[14]:


histone_mark_day0 = np.expand_dims(histone_mark_day0, axis=2)
print histone_mark_day0.shape


# In[15]:


print "Example histone mark signal"
print "\tRaw value: {}".format(bw_histone_mark_day0(normalized_day0_intervals[:1])[0][:5].reshape(-1))


# In[16]:


from keras.layers import Input, Dense, Conv1D, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import optimizers
from keras import metrics
from keras import losses
from keras import backend as K
from keras.callbacks import Callback, TensorBoard, ReduceLROnPlateau, ModelCheckpoint


# In[17]:


dropout_rate = 0.5
# parameters for first conv layer
hidden_filters_1 = 32
hidden_kernel_size_1 = int(window_size*0.6)
# parameters for second conv layer
hidden_filters_2 = 32
hidden_kernel_size_2 = int(window_size*0.25)
# parameters for third conv layer
hidden_filters_3 = 32
hidden_kernel_size_3 = int(window_size*0.15)
# parameters for second conv layer
output_filters = 1
output_kernel_size = 128
# parameters for training
batch_size = 128
num_epochs = 300
evaluation_freq = 10


# In[18]:


'''
Baseline CNN (based on KERAS functional API)
'''
inputs = Input(shape=(window_size, 5, ))
x = Conv1D(
    filters=hidden_filters_1,
    kernel_size=hidden_kernel_size_1,
    padding='same',
    strides=1)(inputs)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(dropout_rate)(x)

x = Conv1D(
    filters=hidden_filters_2,
    kernel_size=hidden_kernel_size_2,
    padding='same',
    strides=1)(inputs)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(dropout_rate)(x)

x = Conv1D(
    filters=hidden_filters_2,
    kernel_size=hidden_kernel_size_2,
    padding='same',
    strides=1)(inputs)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(dropout_rate)(x)

outputs = Conv1D(
    filters=output_filters,
    kernel_size=output_kernel_size,
    padding='same',
    activation='linear',
    strides=1
    )(x)

model = Model(inputs=inputs, outputs=outputs)


# In[19]:


# Print out model summary
print model.summary()


# In[20]:


def pearson(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(np.multiply(xm,ym))
    r_den = K.sqrt(np.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(r)


# In[21]:


# Helper functions for writing the scores into bigwig file
from itertools import izip
from itertools import groupby
import subprocess

def interval_key(interval):
    return (interval.chrom, interval.start, interval.stop)

def merged_scores(scores, intervals, merge_type):
    # A generator that returns merged intervals/scores
    # Scores should have shape: #examples x #categories x #interval_size
    # Second dimension can be omitted for a 1D signal
    signal_dims = scores.ndim - 1
    assert signal_dims in {1, 2}

    # Only support max for now
    assert merge_type == 'max'
    score_first_dim = 1 if signal_dims == 1 else scores.shape[1]

    dtype = scores.dtype

    sort_idx, sorted_intervals =         zip(*sorted(enumerate(intervals),
                    key=lambda item: interval_key(item[1])))
    sorted_intervals = BedTool(sorted_intervals)

    # Require at least 1bp overlap
    # Explicitly convert to list otherwise it will keep opening a file when
    # retrieving an index resulting in an error (too many open files)
    interval_clust = list(sorted_intervals.cluster(d=-1))
    for _, group in groupby(izip(sort_idx, interval_clust),
                            key=lambda item: item[1].fields[-1]):
        idx_interval_pairs = list(group)
        group_idx, group_intervals = zip(*idx_interval_pairs)

        if len(idx_interval_pairs) == 1:
            yield group_intervals[0], scores[group_idx[0], ...]
        else:
            group_chrom = group_intervals[0].chrom
            group_start = min(interval.start for interval in group_intervals)
            group_stop = max(interval.stop for interval in group_intervals)

            # This part needs to change to support more merge_types (e.g. mean)
            group_score = np.full((score_first_dim, group_stop - group_start),
                                  -np.inf, dtype)
            for idx, interval in idx_interval_pairs:
                slice_start = interval.start - group_start
                slice_stop = slice_start + (interval.stop - interval.start)
                group_score[..., slice_start:slice_stop] =                     np.maximum(group_score[..., slice_start:slice_stop],
                               scores[idx, ...])
            if signal_dims == 1:
                group_score = group_score.squeeze(axis=0)
            yield Interval(group_chrom, group_start, group_stop), group_score
            
def interval_score_pairs(intervals, scores, merge_type):
    return (izip(intervals, scores) if merge_type is None
            else merged_scores(scores, intervals, merge_type))

def _write_1D_deeplift_track(scores, intervals, file_prefix, merge_type='max',
                             CHROM_SIZES='/mnt/data/annotations/by_release/hg19.GRCh37/hg19.chrom.sizes'):
    assert scores.ndim == 2

    bedgraph = file_prefix + '.bedGraph'
    bigwig = file_prefix + '.bw'

    print 'Writing 1D track of shape: {}'.format(scores.shape)
    print 'Writing to file: {}'.format(bigwig)

    with open(bedgraph, 'w') as fp:
        for interval, score in interval_score_pairs(intervals, scores,
                                                    merge_type):
            chrom = interval.chrom
            start = interval.start
            for score_idx, val in enumerate(score):
                fp.write('%s\t%d\t%d\t%g\n' % (chrom,
                                               start + score_idx,
                                               start + score_idx + 1,
                                               val))
    print 'Wrote bedgraph.'

    try:
        output = subprocess.check_output(
            ['wigToBigWig', bedgraph, CHROM_SIZES, bigwig],
            stderr=subprocess.STDOUT)
        print 'wigToBigWig output: {}'.format(output)
    except subprocess.CalledProcessError as e:
        print 'wigToBigWig terminated with exit code {}'.format(
            e.returncode)
        print 'output was:\n' + e.output

    print 'Wrote bigwig.'


# In[22]:


model_dir = os.path.join("models", "mse_lowlr_window5000_deep")
log_dir = os.path.join("logs", "mse_lowlr_window5000_deep")
srv_dir = os.path.join("/users", "jesikmin", "mse_lowlr_window5000_deep")
#srv_dir = os.path.join("/srv", "www", "kundaje", "jesikmin", "mse_lowlr_window4000")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(srv_dir):
    os.makedirs(srv_dir)


# In[23]:


# setting an adam optimizer with 1.0 clip norm
#adam = optimizers.Adam(lr=1e-3, clipnorm=1.)
adam = optimizers.SGD(lr=0.01, clipnorm=1.)

print "Compiling a model with adam optimizer"
model.compile(loss=losses.mean_squared_error,
              optimizer=adam,
              metrics=[pearson, metrics.mse, metrics.mae])


# In[24]:


# CallBack: tensorboard with grad histogram
tensorboard = TensorBoard(log_dir=log_dir,
                          histogram_freq=1,
                          batch_size=batch_size,
                          write_grads=True)


# In[25]:


# CallBack: reduce learning rate when validation loss meets plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.96,
                              patience=5,
                              min_lr=1e-7)


# In[26]:


# CallBack: evaluate model for every evaulation epoch
class EvaluateModel(Callback):
    def __init__(self, X_test, y_test):
        self.X_test, self.y_test = X_test, y_test
        self.epochs = 0

    def on_epoch_end(self, batch, logs={}):
        self.epochs += 1
        # evaluate the model
        if self.epochs % evaluation_freq == 0:
            scores = model.evaluate(X_test, y_test)
            print zip(model.metrics_names, scores)
            test_prediction = model.predict(X_test)
            test_spearman_sum = 0.0
            for idx, _pred in enumerate(test_prediction):
                rho, _ = spearmanr(histone_mark_day0[10000+idx], _pred)
                test_spearman_sum += rho
            print "Test spearman: {}".format(test_spearman_sum * 1.0 / 1000)


# In[27]:


# CallBack: store bigwig file for the best model
class SaveBigwig(Callback):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.best_val_loss = float('Inf')
        self.best_epoch = -1
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.epochs = 0
    
    def on_epoch_end(self, batch, logs={}):
        self.epochs += 1
        cur_val_loss = logs['val_loss']
        if cur_val_loss < self.best_val_loss:
            self.best_val_loss = cur_val_loss
            self.best_epoch = self.epochs
            prediction = model.predict(X_train).reshape(7500, window_size)
            _write_1D_deeplift_track(prediction,
                                     normalized_day0_intervals[:7500], os.path.join(srv_dir, 'train'))
            prediction = model.predict(X_val).reshape(2500, window_size)
            _write_1D_deeplift_track(prediction,
                                     normalized_day0_intervals[7500:10000], os.path.join(srv_dir, 'val'))
            prediction = model.predict(X_test).reshape(1000, window_size)
            _write_1D_deeplift_track(prediction,
                                     normalized_day0_intervals[10000:11000], os.path.join(srv_dir, 'test'))
            f = open(os.path.join(srv_dir, 'meta.txt'), 'wb')
            f.write(str(self.epochs) + "  " + str(logs))
            f.close()


# In[28]:


# CallBack: Save model checkpoint based on validation loss
checkpointer = ModelCheckpoint(os.path.join(model_dir, "best_model.h5"),
                               monitor='val_loss',
                               verbose=1,
                               save_best_only=True,
                               mode='min')


# In[29]:


print "Fitting the model..."
X_train, y_train = atac_seq_day0[:7500], histone_mark_day0[:7500]
X_val, y_val = atac_seq_day0[7500:10000], histone_mark_day0[7500:10000]
X_test, y_test = atac_seq_day0[10000:11000], histone_mark_day0[10000:11000]

model.fit(atac_seq_day0[:10000],
          histone_mark_day0[:10000],
          batch_size=batch_size,
          epochs=num_epochs,
          validation_split=0.25,   
          #shuffle=True,
          callbacks=[tensorboard,
                     reduce_lr,
                     EvaluateModel(X_test, y_test),
                     checkpointer,
                     SaveBigwig(X_train, y_train, X_val, y_val, X_test, y_test)])

