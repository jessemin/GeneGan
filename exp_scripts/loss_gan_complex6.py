import os

os.chdir('../exp_notebooks')
os.system('python loss_complex6_gan.py\
          -w=5001\
          -save=loss_gan_complex6\
          -sample_num=10000\
          -g_weight=0.4\
          -mse_weight=1.0\
          -g_lr=0.001\
          -d_lr=0.001\
          -e=1000\
          -cuda=5')
