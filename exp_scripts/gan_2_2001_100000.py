import os

os.chdir('../exp_notebooks')
os.system('python gan_barebone.py\
          -w=2001\
          -save=gan_2_2001_10000\
          -sample_num=100000\
          -cuda=0')
