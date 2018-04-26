'''
Wasserstein GeneGan Jupyter Barebone

Reference: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
Author: Jesik Min
Overview: Barebone Jupyter notebook version of WGAN.
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
from functools import partial
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
                    required=False,
                    type=str,
                    default='',
                    dest="model_path",
                    help="Path to pre-trained model")
parser.add_argument('-save',
                    '--save_dir',
                    required=False,
                    type=str,
                    default='test',
                    dest="save_dir",
                    help="Where to save the best model, logs, and its predictions")
parser.add_argument('-s',
                    '--smooth_rate',
                    required=False,
                    type=float,
                    default=0.0,
                    dest="smooth_rate",
                    help="Smooth rate for training discriminator")
parser.add_argument('-d_freq',
                    '--d_train_freq',
                    required=False,
                    type=int,
                    default=16,
                    dest="d_train_freq",
                    help="Frequency of training discriminator")
parser.add_argument('-sample_num',
                    '--sample_num',
                    required=False,
                    type=int,
                    default=10000,
                    dest="sample_num",
                    help="Total number of train sample; val sample is 0.2*train_num")
parser.add_argument('-cuda',
                    '--cuda',
                    required=True,
                    type=str,
                    dest="cuda",
                    help="Cuda visible devices")
parser.add_argument('-n_critic',
                    '--n_critic',
                    required=False,
                    type=int,
                    default=5,
                    dest="n_critic",
                    help="Number of critics")

args = parser.parse_args()

# Overview of all parameters entered
print "-" * 40
print args
print "-" * 40

# Parse all arguments

window_size = args.window_size
day = args.day
frag = args.frag
histone = args.histone
model_path = args.model_path
save_dir = args.save_dir
sample_num = args.sample_num
cuda = args.cuda
n_critic = args.n_critic

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

# Get train/val/test intervals
train_intervals = list(BedTool(train_dir))
val_intervals = list(BedTool(val_dir))
test_intervals = list(BedTool(test_dir))
print '# of Train Intervals: {}'.format(len(train_intervals))
print '# of Val Intervals: {}'.format(len(val_intervals))
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

# Normalize train intervals
normalized_train_intervals = [normalize_interval(interval, window_size) for interval in train_intervals if normalize_interval(interval, window_size)]
print 'Finished normalizing train intervals!'
# Normalize val intervals
normalized_val_intervals = [normalize_interval(interval, window_size) for interval in val_intervals if normalize_interval(interval, window_size)]
print 'Finished normalizing val intervals!'
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

# Prune train intervals
print "Before pruning train: {}".format(len(normalized_train_intervals))
prune_invalid_intervals(normalized_train_intervals, input_candidates)
print "After pruning train: {}".format(len(normalized_train_intervals))
# Prune val intervals
print "Before pruning val: {}".format(len(normalized_val_intervals))
prune_invalid_intervals(normalized_val_intervals, input_candidates)
print "After pruning val: {}".format(len(normalized_val_intervals))
# Prune test intervals
print "Before pruning test: {}".format(len(normalized_test_intervals))
prune_invalid_intervals(normalized_test_intervals, input_candidates)
print "After pruning test: {}".format(len(normalized_test_intervals))


X_train = input_candidates(normalized_train_intervals)
print "Finished fetching X_train"
X_val = input_candidates(normalized_val_intervals)
print "Finished fetching X_val"
X_test = input_candidates(normalized_test_intervals)
print "Finished fetching X_test"
print X_train.shape, X_val.shape, X_test.shape

print "Dimension of ATAC-seq signal (input): {}".format(X_train[0].shape)

# Replace nan values with zeros
y_train = np.nan_to_num(output_candidates(normalized_train_intervals))
print "Finished fetching y_train"
y_val = np.nan_to_num(output_candidates(normalized_val_intervals))
print "Finished fetching y_val"
y_test = np.nan_to_num(output_candidates(normalized_test_intervals))
print "Finished fetching y_test"
print y_train.shape, y_val.shape, y_test.shape


y_train = np.expand_dims(y_train, axis=2)
y_val = np.expand_dims(y_val, axis=2)
y_test = np.expand_dims(y_test, axis=2)
print y_train.shape, y_val.shape, y_test.shape

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
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.merge import _Merge


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

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

# WGAN Model (based on WGan)
class GAN():
    def __init__(self, window_size,
                 X_train, y_train,
                 X_val, y_val,
                 X_test, y_test,
                 model_dir, srv_dir):

        # Set train/val/test
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
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
        # 4) WGAN-specific parameters
        self.n_critic = n_critic

        # Build and compile the discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        for layer in self.discriminator.layers:
            layer.trainable = False
        self.discriminator.trainable = False

        # The generator takes noise as input and generated imgs
        generator_input = Input(shape=self.input_shape)
        generator_layers = self.generator(generator_input)
        discriminator_layers_for_generator = self.discriminator(generator_layers)
        self.generator_model = Model(inputs=[generator_input],
                                     outputs=[discriminator_layers_for_generator])
        # We use the Adam paramaters from Gulrajani et al.
        self.generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                                               loss=self.wasserstein_loss)

        # Now that the generator_model is compiled, we can make the discriminator layers trainable.
        for layer in self.discriminator.layers:
            layer.trainable = True
        for layer in self.generator.layers:
            layer.trainable = False
        self.discriminator.trainable = True
        self.generator.trainable = False

        real_samples = Input(shape=self.output_shape)
        generator_input_for_discriminator = Input(shape=self.input_shape)
        generated_samples_for_discriminator = self.generator(generator_input_for_discriminator)
        discriminator_output_from_generator = self.discriminator(generated_samples_for_discriminator)
        discriminator_output_from_real_samples = self.discriminator(real_samples)

        # We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
        averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
        # We then run these samples through the discriminator as well. Note that we never really use the discriminator
        # output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
        averaged_samples_out = self.discriminator(averaged_samples)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=10)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                                         outputs=[discriminator_output_from_real_samples,
                                                  discriminator_output_from_generator,
                                                  averaged_samples_out])
        # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
        # samples, and the gradient penalty loss for the averaged samples.
        self.discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                                         loss=[self.wasserstein_loss,
                                               self.wasserstein_loss,
                                               partial_gp_loss])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples, gradient_penalty_weight):
        """ Computes gradient penalty based on prediction and weighted real / fake samples """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        # ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        # ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def build_generator(self):
        # Generator
        # 1) 32 * window_size Conv1D layers with RELU and Dropout
        model = Sequential()

        model.add(Conv1D(hidden_filters_1,
                         2200,
                         padding="same",
                         strides=1,
                         input_shape=self.input_shape,
                         activation='relu',
                         name='gen_conv1d_1'))
        model.add(Dropout(dropout_rate,
                  name='gen_dropout_1'))

        model.add(Conv1D(hidden_filters_1,
                         1200,
                         padding="same",
                         strides=1,
                         input_shape=self.input_shape,
                         activation='relu',
                         name='gen_conv1d_2'))
        model.add(Dropout(dropout_rate,
                  name='gen_dropout_2'))

        model.add(Conv1D(hidden_filters_1,
                         800,
                         padding="same",
                         strides=1,
                         input_shape=self.input_shape,
                         activation='relu',
                         name='gen_conv1d_3'))
        model.add(Dropout(dropout_rate,
                  name='gen_dropout_3'))

        model.add(Conv1D(hidden_filters_1,
                         500,
                         padding="same",
                         strides=1,
                         input_shape=self.input_shape,
                         activation='relu',
                         name='gen_conv1d_4'))
        model.add(Dropout(dropout_rate,
                  name='gen_dropout_4'))

        model.add(Conv1D(hidden_filters_1,
                         300,
                         padding="same",
                         strides=1,
                         input_shape=self.input_shape,
                         activation='relu',
                         name='gen_conv1d_5'))
        model.add(Dropout(dropout_rate,
                  name='gen_dropout_5'))

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

        # load weights for generator if specified
        if model_path:
            print "-"*50
            print model.get_weights()
            model.load_weights(model_path, by_name=True)
            print "-"*50
            print model.get_weights()

        return model

    def build_discriminator(self):
        # Discriminator
        # 1) 16 * 200 Conv1D with LeakyRelu, Dropout
        model = Sequential()

        model.add(Conv1D(hidden_filters_1,
                         200,
                         padding="valid",
                         strides=1,
                         kernel_initializer='he_normal',
                         input_shape=self.output_shape))
        model.add(LeakyReLU())

        # 2) Average Pooling, Flatten, Dense, and LeakyRelu
        model.add(AveragePooling1D(25))
        model.add(Flatten())
        model.add(Dense(int(window_size/16),
                        kernel_initializer='he_normal'))
        model.add(LeakyReLU())

        # 3) Final output with no activation
        model.add(Dense(1, kernel_initializer="he_normal"))

        print "Discriminator"
        model.summary()

        return model

    def train(self, epochs, batch_size):
        d_loss_history, g_loss_history = [], []
        pearson_train_history, pearson_val_history = [], []

        max_pearson = -1.0

        # size of the half of the batch
        half_batch = int(batch_size / 2)
        d_loss_real, d_loss_fake, g_loss = [1, 0], [1, 0], [1, 0]

        positive_y = np.ones((batch_size, 1),
                             dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((batch_size, 1),
                           dtype=np.float32)

        for epoch in range(epochs):
            # list for storing losses/accuracies for both discriminator and generator
            d_losses, d_accuracies, g_losses = [], [], []

            for _minibatch_idx in range(int(sample_num/batch_size)):
                for _ in range(self.n_critic):
                    dis_idx = np.random.randint(0, y_train.shape[0], batch_size)
                    discriminator_minibatches = y_train[dis_idx]
                    noise = self.X_train[dis_idx].astype(np.float32)
                    d_loss = self.discriminator_model.train_on_batch([discriminator_minibatches, noise],
                                                                     [positive_y, negative_y, dummy_y])
                    d_losses.append(d_loss)
                gen_idx = np.random.randint(0, y_train.shape[0], batch_size)
                noise = self.X_train[gen_idx].astype(np.float32)
                g_losses.append(self.generator_model.train_on_batch(noise,
                                                                    positive_y))

            # ---------------------
            # Convert each histories into numpy arrays to get means
            # ---------------------
            d_losses = np.array(d_losses)
            d_accuracies = np.array(d_accuracies)
            g_losses = np.array(g_losses)

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

            # if current pearson on validation set is greatest so far, update the max pearson,
            if max_pearson < avg_val_pearson:
                print "Perason on val improved from {} to {}".format(max_pearson, avg_val_pearson)
                _write_1D_deeplift_track(predictions.reshape(self.X_train.shape[0], self.window_size),
                                         normalized_train_intervals, os.path.join(self.srv_dir, 'train'))
                _write_1D_deeplift_track(val_predictions.reshape(self.X_val.shape[0], self.window_size),
                                         normalized_val_intervals, os.path.join(self.srv_dir, 'val'))
                f = open(os.path.join(self.srv_dir, 'meta.txt'), 'wb')
                f.write(str(epoch) + " " + str(avg_pearson) + "  " + str(avg_val_pearson) + "\n")
                max_pearson = avg_val_pearson

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

                self.generator.save(os.path.join(self.model_dir, 'best_generator.h5'))
                self.discriminator.save(os.path.join(self.model_dir, 'best_discriminator.h5'))

            # Save the progress
            d_loss_history.append(d_losses)
            g_loss_history.append(g_losses)
            pearson_train_history.append(avg_pearson)
            pearson_val_history.append(avg_val_pearson)

            # Print the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_losses.mean(), 100.0*d_accuracies.mean(), g_losses.mean()))

        assert (len(d_loss_history) == len(g_loss_history) == len(pearson_train_history) == len(pearson_val_history))

        print "Saving the loss and pearson logs..."
        np.save(os.path.join(log_dir, 'd_loss_history.npy'), d_loss_history)
        np.save(os.path.join(log_dir, 'g_loss_history.npy'), g_loss_history)
        np.save(os.path.join(log_dir, 'pearson_train_history.npy'), pearson_train_history)
        np.save(os.path.join(log_dir, 'pearson_val_history.npy'), pearson_val_history)
        print "Train Complete!"


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
gan.train(num_epochs, batch_size)

