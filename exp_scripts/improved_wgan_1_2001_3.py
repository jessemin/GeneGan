import os

os.chdir('../exp_notebooks')
os.system('python improved_wgan.py\
          -w=2001\
          -save=improved_wgan_1_2001_3\
          -sample_num=10000\
          -n_critic=3\
          -cuda=5')
