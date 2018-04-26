import os

os.chdir('../exp_notebooks')
os.system('python loss_improved_wgan.py\
          -w=2001\
          -save=loss_improved_wgan\
          -sample_num=10000\
          -n_critic=3\
          -m=/users/jesikmin/GeneGan/exp_notebooks/models/cnn_2001/best_model.h5\
          -cuda=6')
