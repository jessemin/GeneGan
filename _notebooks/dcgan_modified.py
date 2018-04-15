
# coding: utf-8

# # DCGan Baseline

# #### NOTE: Need to activate genomelake environment before this code. Simply type 'genomelake' in terminal.

# In[1]:


#get_ipython().magic(u'env CUDA_VISIBLE_DEVICES=5,6,7')
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


window_size = 2001
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


from keras.layers import Input, Dense, Conv1D, Dropout, BatchNormalization, Activation, ZeroPadding1D, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras import optimizers
from keras import metrics
from keras import losses
from keras import backend as K
from keras.callbacks import Callback, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam


# In[17]:


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


# In[18]:


dropout_rate = 0.5
# parameters for first conv layer
hidden_filters_1 = 32
hidden_kernel_size_1 = 2001
# parameters for second conv layer
output_filters = 1
output_kernel_size = 16
# parameters for training
batch_size = 128
num_epochs = 300
evaluation_freq = 10


# In[31]:


class GAN():
    def __init__(self):
        self.window_size = window_size
        self.channels = 5
        self.input_shape = (self.window_size, self.channels,)
        self.output_shape = (self.window_size, 1,)

        #optimizer = Adam(lr=0.0002, clipnorm=1., beta_1=0.5)
        optimizer = Adam(lr=0.001, clipnorm=1.)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
                                   optimizer=optimizer,
                                   metrics=['accuracy', metrics.mse])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=self.input_shape)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        print self.combined.summary()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = self.input_shape
        
        model = Sequential()
        
        model.add(Conv1D(32, 2001, padding="same", strides=1, input_shape=noise_shape, activation='relu'))
        model.add(Dropout(0.5))
        
        model.add(Conv1D(1, 16, padding='same', strides=1))
        model.add(Activation('linear'))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = self.output_shape
        
        model = Sequential()
        
        model.add(Conv1D(32, 1001, padding="same", strides=1, input_shape=img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(32, 600, padding="same", strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(32, 300, padding="same", strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(32, 100, padding="same", strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        
        model.add(Flatten())
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)
        
        return Model(img, validity)

    def train(self, epochs, batch_size, X_train, y_train):

        half_batch = int(batch_size / 2)
            
        d_loss_real, d_loss_fake, g_loss = [1, 0], [1, 0], [1, 0]
        
        max_pearson = -1.0
            
        for epoch in range(epochs):
            
            # list for storing losses/accuracies for both discriminator and generator
            d_losses, d_accuracies, g_losses = [], [], []
            
            # sufficient number of minibatches for each epoch
            for _minibatch_idx in range(128):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                trained_discriminator = False
                # Select a random half batch of images
                dis_idx = np.random.choice(y_train.shape[0], half_batch, replace=False)
                imgs = y_train[dis_idx]
                dis_noise = X_train[dis_idx]

                # Generate a half batch of new images
                gen_imgs = self.generator.predict(dis_noise)

                # Train the discriminator with label smoothing
                smoothed_idx = np.random.choice(half_batch, int(half_batch*0.5), replace=False)
                smoothed_labels = np.ones((half_batch, 1))
                smoothed_labels[smoothed_idx] = 0
                
                #if d_loss_real[0] > 0.45 or g_loss < 0.1:
                #    d_loss_real = self.discriminator.train_on_batch(imgs, smoothed_labels)
                #    trained_discriminator = True
                #if d_loss_fake[0] > 0.6 or g_loss < 0.1:
                #    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
                #    trained_discriminator = True
                    
                # take the average of each loss and accuracy
                #if trained_discriminator:
                #    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_loss_real = self.discriminator.train_on_batch(imgs, smoothed_labels)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # ---------------------
                #  Train Generator
                # ---------------------
                
                gen_idx = np.random.randint(0, y_train.shape[0], batch_size)
                gen_noise = X_train[gen_idx]

                # The generator wants the discriminator to label the generated samples
                # as valid (ones)
                valid_y = np.array([1] * batch_size)

                # Train the generator
                g_loss = self.combined.train_on_batch(gen_noise, valid_y)

                #if trained_discriminator:
                d_losses.append(d_loss[0])
                d_accuracies.append(d_loss[1])
                
                g_losses.append(g_loss)
                
            # convert each histories into numpy arrays to get means
            d_losses = np.array(d_losses)
            d_accuracies = np.array(d_accuracies)
            g_losses = np.array(g_losses)
            
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_losses.mean(), 100.0*d_accuracies.mean(), g_losses.mean()))
            
            predictions = self.generator.predict(X_train)
            pearsons = []
            for pred_idx in range(len(predictions)):
                prediction = predictions[pred_idx]
                pearsons.append(pearsonr(prediction, y_train[pred_idx]))
            avg_pearson = np.array(pearsons).mean()
            print "Pearson R on Train set: {}".format(avg_pearson)
            
            val_predictions = self.generator.predict(X_val)
            val_pearsons = []
            for val_pred_idx in range(len(val_predictions)):
                prediction = val_predictions[val_pred_idx]
                val_pearsons.append(pearsonr(prediction, y_val[val_pred_idx]))
            avg_val_pearson = np.array(val_pearsons).mean()
            print "Pearson R on Val set: {}".format(avg_val_pearson)  
            
            if max_pearson < avg_val_pearson:
                print "Perason on val improved from {} to {}".format(max_pearson, avg_pearson)
                _write_1D_deeplift_track(predictions.reshape(7500, 2001),
                                         normalized_day0_intervals[:7500], os.path.join(srv_dir, 'train'))
                _write_1D_deeplift_track(val_predictions.reshape(2500, 2001),
                                         normalized_day0_intervals[7500:10000], os.path.join(srv_dir, 'val'))
                f = open(os.path.join(srv_dir, 'meta.txt'), 'wb')
                f.write(str(epoch) + "  " + str(avg_val_pearson))
                f.close()
                max_pearson = avg_val_pearson


# In[32]:


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


# In[33]:


model_dir = os.path.join("models", "dcgan_mod")
log_dir = os.path.join("logs", "dcgan_mod")
srv_dir = os.path.join("/users", "jesikmin", "dcgan_mod")
#srv_dir = os.path.join("/srv", "www", "kundaje", "jesikmin", "gan")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(srv_dir):
    os.makedirs(srv_dir)


# In[34]:


print "Fitting the model..."
X_train, y_train = atac_seq_day0[:7500], histone_mark_day0[:7500]
X_val, y_val = atac_seq_day0[7500:10000], histone_mark_day0[7500:10000]
X_test, y_test = atac_seq_day0[10000:11000], histone_mark_day0[10000:11000]

gan = GAN()
gan.train(num_epochs, batch_size, X_train, y_train)

