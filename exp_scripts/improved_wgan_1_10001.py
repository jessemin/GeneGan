import os

os.chdir('../exp_notebooks')
os.system('python improved_wgan.py\
          -w=10001\
          -save=improved_wgan_1_10001\
          -sample_num=10000\
          -cuda=6')
