import os

os.chdir('../exp_notebooks')
os.system('python gan_final.py\
          -w=2001\
          -save=gan_on_2001\
          -sample_num=10000\
          -g_weight=0.4\
          -mse_weight=1.0\
          -g_lr=0.0005\
          -d_lr=0.0001\
          -e=1000\
          --smooth_rate=0.1\
          -cuda=1')
