import os

os.chdir('../exp_notebooks')
os.system('python cnn.py\
          -w=2001\
          -save=cnn_2001_sample\
          -sample_num=100000\
          -cuda=5')
