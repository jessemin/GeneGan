import os

os.chdir('../exp_notebooks')
os.system('python gan_final.py\
          -w=10001\
          -save=gan_final\
          -sample_num=100000\
          -g_weight=0.4\
          -mse_weight=1.0\
          -g_lr=0.0005\
          -d_lr=0.0001\
          -e=1000\
          --smooth_rate=0.01\
          -cuda=7')
