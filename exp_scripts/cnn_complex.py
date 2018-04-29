import os

os.chdir('../exp_notebooks')
os.system('python cnn_complex.py\
          -w=5001\
          -save=cnn_complex\
          -sample_num=10000\
          -cuda=5')
