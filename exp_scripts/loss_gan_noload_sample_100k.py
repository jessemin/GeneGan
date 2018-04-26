import os

os.chdir('../exp_notebooks')
os.system('python loss_gan.py\
          -w=2001\
          -save=loss_gan_noload_sample_100k\
          -sample_num=100000\
          -g_weight=0.4\
          -mse_weight=1.0\
          -g_lr=0.0005\
          -d_lr=0.0001\
          -d_freq=16\
          --smooth_rate=0.01\
          -cuda=0')
