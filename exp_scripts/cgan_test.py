import os

os.chdir('../exp_notebooks')
os.system('python cgan.py\
          -w=2001\
          -save=cgan_test\
          -sample_num=10000\
          -cuda=5')
