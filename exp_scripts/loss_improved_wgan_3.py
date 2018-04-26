import os

os.chdir('../exp_notebooks')
os.system('python loss_improved_wgan.py\
          -w=2001\
          -save=loss_improved_wgan_3\
          -sample_num=10000\
          -n_critic=1\
          -d_freq=32\
          -w_weight=0.08\
          -mse_weight=1.0\
          -g_lr=0.001\
          -d_lr=0.00001\
          -m=/users/jesikmin/GeneGan/exp_notebooks/models/cnn_2001/best_model.h5\
          -cuda=0')
