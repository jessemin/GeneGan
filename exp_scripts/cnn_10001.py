import os

os.chdir('../exp_notebooks')
os.system('python cnn.py\
          -w=10001\
          -save=cnn_10001\
          -sample_num=10000\
          -cuda=1')
