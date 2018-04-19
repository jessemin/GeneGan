import os

os.chdir('../exp_notebooks')
os.system('python wgan.py\
          -w=2001\
          -d_freq=1\
          -save=wgan_test\
          -sample_num=10000\
          -cuda=1')
