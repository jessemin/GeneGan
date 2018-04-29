import os

os.chdir('../exp_notebooks')
os.system('python loss_complex3_gan.py\
          -w=2001\
          -save=loss_gan_complex3\
          -sample_num=50000\
          -g_weight=0.4\
          -mse_weight=1.0\
          -g_lr=0.001\
          -d_lr=0.001\
          -e=1000\
          -d_freq=1\
          --smooth_rate=0.1\
          -cuda=4')
