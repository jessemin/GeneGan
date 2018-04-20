import os

os.chdir('../exp_notebooks')
os.system('python improved_wgan.py\
          -w=2001\
          -save=improved_wgan_2_2001_100000\
          -sample_num=100000\
          -cuda=0,1')
