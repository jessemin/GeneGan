import os

os.chdir('../exp_notebooks')
os.system('python improved_wgan.py\
          -w=2001\
          -save=improved_wgan_1_2001_3_load\
          -sample_num=10000\
          -n_critic=3\
          -m=/users/jesikmin/GeneGan/_notebooks/models/cnn_2000_new/best_model.h5\
          -cuda=4')
