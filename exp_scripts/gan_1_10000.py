import os

os.chdir('../exp_notebooks')
os.system('python gan_barebone.py\
          -w=2001\
          -save_m=gan_1_10000\
          -save_srv=gan_1_10000\
          -sample_num=10000\
          -cuda=0')
