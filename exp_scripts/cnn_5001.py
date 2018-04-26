import os

os.chdir('../exp_notebooks')
os.system('python cnn.py\
          -w=5001\
          -save=cnn_5001\
          -sample_num=10000\
          -cuda=7')
