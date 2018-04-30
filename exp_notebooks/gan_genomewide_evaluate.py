'''
GeneGan Jupyter Barebone

Author: Jesik Min
Overview: Barebone Jupyter notebook version of vanilla-GAN.
Note: Need to activate genomelake environment before running this code. Simply type 'genomelake' in terminal on kali under 'jesikmin'.
'''

import os, sys
sys.path.append("..")
# Custom file path package
from data import Data_Directories
# Custom utility package
from utils.compute_util import *
# Package for genomic data
from pybedtools import Interval, BedTool
from genomelake.extractors import ArrayExtractor, BigwigExtractor
# Package for plotting
import matplotlib.pyplot as plt
# Package for correlation
from scipy.stats.stats import pearsonr,spearmanr
# Tensorflow
import tensorflow as tf
# ArgParsing
import argparse
parser = argparse.ArgumentParser()

# Setup Arguments
parser.add_argument('-e',
                    '--num_epochs',
                    required=False,
                    type=int,
                    default=300,
                    dest="num_epochs",
                    help="Number of epochs for training")
parser.add_argument('-w',
                    '--window_size',
                    required=False,
                    type=int,
                    default=10001,
                    dest="window_size",
                    help="Window size for normalized intervals")
parser.add_argument('-d',
                    '--day',
                    required=False,
                    type=str,
                    default='day0',
                    dest="day",
                    help="Target day of data")
parser.add_argument('-f',
                    '--frag',
                    required=False,
                    type=str,
                    default='140',
                    dest="frag",
                    help="Target fragment length of data")
parser.add_argument('-o',
                    '--output',
                    required=False,
                    type=str,
                    default='H3K27ac',
                    dest="histone",
                    help="Target histone mark of data")
parser.add_argument('-m',
                    '--model_path',
                    type=str,
                    default='/users/jesikmin/GeneGan/exp_notebooks/models/gan_final_2/best_generator.h5',
                    dest="model_path",
                    help="Path to pre-trained model")
parser.add_argument('-save',
                    '--save_dir',
                    required=False,
                    type=str,
                    default='test',
                    dest="save_dir",
                    help="Where to save the best model, logs, and its predictions")
parser.add_argument('-cuda',
                    '--cuda',
                    required=True,
                    type=str,
                    dest="cuda",
                    help="Cuda visible devices")



args = parser.parse_args()

# Parse all arguments

window_size = args.window_size
day = args.day
frag = args.frag
histone = args.histone
model_path = args.model_path
save_dir = args.save_dir
cuda = args.cuda

os.environ["CUDA_VISIBLE_DEVICES"]=cuda

# Logging directories
model_dir = os.path.join("models", save_dir)
log_dir = os.path.join("logs", save_dir)
srv_dir = os.path.join("/srv", "www", "kundaje", "jesikmin", "experiments", save_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(srv_dir):
    os.makedirs(srv_dir)

# Train/val/test intervals
DATA_DIR = '/srv/scratch/jesikmin'
train_dir, val_dir, test_dir = os.path.join(DATA_DIR, 'train_interval'),\
                               os.path.join(DATA_DIR, 'val_interval'),\
                               os.path.join(DATA_DIR, 'test_interval')

print train_dir, val_dir, test_dir

# Get test interval
test_intervals = list(BedTool(test_dir))
print '# of Test Intervals: {}'.format(len(test_intervals))

# Get input/output data directories
data = Data_Directories()
print data.intervals.keys()
print data.input_atac[day].keys()
print data.output_histone[day].keys()

# Extract input candidates
# Create an ArrayExtractor for ATAC-seq of a given day and specified fragment length
input_candidates = ArrayExtractor(data.input_atac[day][frag])
print 'Finished extracting bigwig for {}, {}bp'.format(day, frag)

# Extract output candiates
# Create a BigWigExtractor for histone mark of a given day
output_candidates = BigwigExtractor(data.output_histone[day][histone])
print 'Finished extracting bigwig for {}, {}'.format(day, histone)

# Normalize test intervals
normalized_test_intervals = [normalize_interval(interval, window_size) for interval in test_intervals if normalize_interval(interval, window_size)]
print 'Finished normalizing test intervals!'

# Fetch intervals of sample_num
normalized_train_intervals = normalized_train_intervals[:sample_num]
normalized_val_intervals = normalized_val_intervals[:int(sample_num*0.2)]
print 'Finished fethcing {} train set and {} val set'.format(sample_num, int(sample_num*0.2))

# Assertions of normalization step
assert (sample_num==len(normalized_train_intervals))
assert (int(sample_num*0.2)==len(normalized_val_intervals))
assert (len(test_intervals)==len(normalized_test_intervals))
# Examples of normalized intervals
print "Examples of original train intervals"
print [(int(_interval.start)+int(_interval[-1]), [int(_interval.start), int(_interval.end)])
       for _interval in train_intervals[:3]]
print "Examples of normalized train intervals with window size of {}".format(window_size)
print [([int(_interval.start), int(_interval.end)])
       for _interval in  normalized_train_intervals[:3]]

# Prune intervals that don's make sense
def prune_invalid_intervals(intervals, bigwig_file):
    for _interval in intervals[:]:
        try:
            bigwig_file([_interval])
        except:
            intervals.remove(_interval)
            pass

# Prune test intervals
print "Before pruning test: {}".format(len(normalized_test_intervals))
prune_invalid_intervals(normalized_test_intervals, input_candidates)
print "After pruning test: {}".format(len(normalized_test_intervals))

X_test = input_candidates(normalized_test_intervals)
print "Finished fetching X_test"
print X_test.shape

print "Dimension of ATAC-seq signal (input): {}".format(X_train[0].shape)

# Replace nan values with zeros

y_test = np.nan_to_num(output_candidates(normalized_test_intervals))
print "Finished fetching y_test"
print y_test.shape

y_test = np.expand_dims(y_test, axis=2)
print y_test.shape

print "Dimension of histone mark signal (output): {}".format(y_train[0].shape)

'''
Generative Adversarial Model for Genomics Seq-to-Seq
'''
# Import keras
from keras.layers import AveragePooling1D, Input, Dense, Conv1D, Dropout, BatchNormalization, Activation, ZeroPadding1D, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras import optimizers
from keras import metrics
from keras import losses
from keras import backend as K
from keras.callbacks import Callback, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam, SGD


'''
HYPERPARAMETERS
'''
# GAN Discriminator
smooth_rate = args.smooth_rate
d_train_freq = args.d_train_freq
# Dropout Rate
dropout_rate = 0.5
# First conv layer
hidden_filters_1 = 32
hidden_kernel_size_1 = window_size
# Second conv layer
output_filters = 1
output_kernel_size = 32
# Training
batch_size = 128
num_epochs = args.num_epochs


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
                group_score[..., slice_start:slice_stop] = np.maximum(group_score[..., slice_start:slice_stop], scores[idx, ...])
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



# GAN Model (based on vanilla gan)
class GAN():
    def __init__(self, window_size,
                 X_test, y_test,
                 model_dir, srv_dir):

        # Set train/val/test
        self.X_test, self.y_test = X_test, y_test

        self.model_dir = model_dir
        self.srv_dir = srv_dir

        # Basic parameters
        # 1) Window size of input/ouput signal
        self.window_size = window_size
        # 2) Number of channels
        self.channels = 5
        # 3) Input and Output shape
        self.input_shape = (self.window_size, self.channels,)
        self.output_shape = (self.window_size, 1,)

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=optimizer)

    def build_generator(self):
        # Generator
        # 1) 32 * window_size Conv1D layers with RELU and Dropout

        noise_shape = self.input_shape

        model = Sequential()

        model.add(Conv1D(hidden_filters_1,
                         128,
                         padding="same",
                         strides=1,
                         input_shape=(None, 5),
                         activation='relu',
                         dilation_rate=10,
                         name='gen_conv1d_1'))
        model.add(Dropout(dropout_rate,
                  name='gen_dropout_1'))

        model.add(Conv1D(hidden_filters_1,
                         256,
                         padding="same",
                         strides=1,
                         activation='relu',
                         dilation_rate=5,
                         name='gen_conv1d_2'))
        model.add(Dropout(dropout_rate,
                  name='gen_dropout_2'))

        model.add(Conv1D(hidden_filters_1,
                         128,
                         padding="same",
                         strides=1,
                         dilation_rate=1,
                         activation='relu',
                         name='gen_conv1d_3'))
        model.add(Dropout(dropout_rate,
                  name='gen_dropout_3'))

        # 2) 1 * 16 Conv1D layers with Linear
        # NOTE: All same padding
        model.add(Conv1D(output_filters,
                         output_kernel_size,
                         padding='same',
                         strides=1,
                         activation='linear',
                         name='gen_conv1d_output'))

        print "Generator"
        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        # load weights for generator if specified
        if model_path:
            print "-"*50
            print model.get_weights()
            model.load_weights(model_path, by_name=True)
            print "-"*50
            print model.get_weights()

        return Model(noise, img)

    def evaluate(self):
        # ---------------------
        # Get generator's prediction and compute overall pearson on train set
        # ---------------------
        predictions = self.generator.predict(self.X_train).flatten()
        avg_pearson = pearsonr(predictions, self.y_train.flatten())[0]
        print "Pearson R on Train set: {}".format(avg_pearson)

        # ---------------------
        # Get generator's prediction and compute overall pearson on validation set
        # ---------------------
        val_predictions = self.generator.predict(self.X_val).flatten()
        avg_val_pearson = pearsonr(val_predictions, self.y_val.flatten())[0]
        print "Pearson R on Val set: {}".format(avg_val_pearson)

        f = open(os.path.join(self.srv_dir, 'meta.txt'), 'wb')
        f.write(str(epoch) + " " + str(avg_pearson) + "  " + str(avg_val_pearson) + "\n")

        # ---------------------
        # Get generator's prediction and compute overall pearson on test set
        # ---------------------
        test_predictions = self.generator.predict(self.X_test).flatten()
        avg_test_pearson = pearsonr(test_predictions, self.y_test.flatten())
        print "Pearson R on Test set: {}".format(avg_test_pearson)

        f.write("Test Pearson: " + str(avg_test_pearson))
        f.close()
        _write_1D_deeplift_track(test_predictions.reshape(self.X_test.shape[0], self.window_size),
                                 normalized_test_intervals, os.path.join(self.srv_dir, 'test'))


        print "Evaluation Complete!"


# Helper function for computing Pearson R in Keras
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


print "Training the model..."
gan = GAN(window_size,
          X_train, y_train,
          X_val, y_val,
          X_test, y_test,
          model_dir, srv_dir)
gan.evaluate()

