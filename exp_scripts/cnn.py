import os

os.chdir('../exp_notebooks')
os.system('python cnn.py\
          -w=2001\
          -save=cnn_2001_2\
          -sample_num=10000\
          -cuda=0')
