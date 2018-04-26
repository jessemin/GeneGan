import os

os.chdir('../exp_notebooks')
os.system('python new_improved_gan.py\
          -w=2001\
          -d_freq=32\
          -save=new_improved_wgan\
          -sample_num=10000\
          -cuda=1')
