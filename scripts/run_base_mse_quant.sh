#!/bin/bash
/users/jesikmin/anaconda2/envs/genomelake_env/bin/python gene_conv_simple.py --all --output_dir='Plots_base_mse' --model_dir='Models_base_mse' --output_norm_scheme=quant --batch_size=128 --window_size=4001
