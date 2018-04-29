import os

os.chdir('../exp_notebooks')
os.system('python cnn_final.py\
          -w=10001\
          -save=cnn_final\
          -sample_num=100000\
          -cuda=0,4')
