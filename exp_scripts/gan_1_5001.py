import os

os.chdir('../exp_notebooks')
os.system('python gan_barebone.py\
          -w=5001\
          -save=gan_1_5001\
          -sample_num=10000\
          -cuda=0')
