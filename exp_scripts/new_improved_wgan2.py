import os

os.chdir('../exp_notebooks')
os.system('python new_improved_gan2.py\
          -w=2001\
          -save=new_improved_wgan2\
          -sample_num=10000\
          -cuda=4')
