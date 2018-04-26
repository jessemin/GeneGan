import os

os.chdir('../exp_notebooks')
os.system('python loss_gan.py\
          -w=2001\
          -save=loss_gan\
          -sample_num=10000\
          -g_weight=0.4\
          -mse_weight=1.0\
          -g_lr=0.0005\
          -d_lr=0.0001\
          -d_freq=16\
          --smooth_rate=0.1\
          -m=/users/jesikmin/GeneGan/exp_notebooks/models/cnn_2001_2/best_model.h5\
          -cuda=4')
