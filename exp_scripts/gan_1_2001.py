import os

os.chdir('../exp_notebooks')
os.system('python gan_barebone.py\
          -w=2001\
          -save=gan_1_2001\
          -sample_num=10000\
          -cuda=0')
