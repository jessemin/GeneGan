import os

os.chdir('../exp_notebooks')
os.system('python new_improved_gan.py\
          -w=5001\
          -save=new2_improved_wgan\
          -sample_num=10000\
          -cuda=5')
