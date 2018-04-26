import os

os.chdir('../exp_notebooks')
os.system('python clip_improved_gan.py\
          -w=10001\
          -save=clip_improved_wgan_1_10001\
          -sample_num=10000\
          -d_freq=16\
          -n_critic=3\
          -cuda=1')
