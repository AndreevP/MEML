import os
import json
import time
import datetime
import argparse
os.chdir("./MEF")

import numpy as np

import sys
import torchvision
import torch
from MEF.main import main
from copy import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MEF')
    parser.add_argument('--mode', type=str, default='sample',
                        choices=['train', 'test', 'sample'],
                        help='mode')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='save directory (default is "MEF/results"), when using mode "test" or "sample" script expects pretrained weights to be placed at save_dir/models')
    parser.add_argument('--matrix_exp', type=str, default='default',
                        choices=['default', 'optimized_taylor', 'pade', 'second_limit', 'pade3'],
                        help='method for computing of matrix exponential')    
    parser.add_argument('--matmuls', type=int, default=0,
                        help='the maximal number of matmuls to compute matrix exponential (0 if it is not limited)') 
    parse_args = parser.parse_args()
    
    D = copy(parse_args.__dict__)

    param_dir = './config.json'
    if param_dir:
        with open(param_dir) as f_obj:
            parse_args.__dict__ = json.load(f_obj)
    for arg in ["mode", "device", "save_dir", "matrix_exp"]:
        if arg == "matrix_exp":
            v = D[arg] + " " + str(D["matmuls"])
        else:
            v = D[arg]
        parse_args.__dict__[arg] = v
    
    print(parse_args)

    np.random.seed(parse_args.seed)
    torch.manual_seed(parse_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(parse_args.seed)
    main(parse_args)
