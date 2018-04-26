import os

os.chdir('../exp_notebooks')
os.system('python loss_gan.py\
          -w=2001\
          -save=loss_gan_noload_3\
          -sample_num=10000\
          -g_weight=0.5\
          -mse_weight=0.5\
          -g_lr=0.001\
          -d_lr=0.001\
          --smooth_rate=0.1\
          -cuda=1')
