import os

os.chdir('../exp_notebooks')
os.system('python improved_wgan.py\
          -w=2001\
          -save=improved_wgan_1_2001\
          -sample_num=10000\
          -cuda=0')
