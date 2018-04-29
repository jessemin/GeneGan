import os

os.chdir('../exp_notebooks')
os.system('python loss_improved_wgan_new.py\
          -w=2001\
          -save=loss_improved_wgan_new\
          -sample_num=10000\
          -n_critic=5\
          -w_weight=0.5\
          -mse_weight=1.0\
          -g_lr=0.0001\
          -d_lr=0.0001\
          -s=0.1\
          -cuda=5')
