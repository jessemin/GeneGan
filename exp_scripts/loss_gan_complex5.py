import os

os.chdir('../exp_notebooks')
os.system('python loss_complex5_gan.py\
          -w=10001\
          -save=loss_gan_complex5_3_10001\
          -sample_num=10000\
          -g_weight=0.4\
          -mse_weight=1.0\
          -g_lr=0.0005\
          -d_lr=0.0001\
          -e=1000\
          --smooth_rate=0.01\
          -cuda=4')
