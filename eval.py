"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.
Description :
Author：Team Li
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from utils.config import get_eval_config
from utils.logger import logger
from utils.utils import exists,list_files
from trainers.base_evaluate import Evaluater

def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        assert opts.batch_size > 0

def main():
    parser = get_eval_config()
    opts = parser.parse_args()
    check_opts(opts)

    evaluater = Evaluater()

    logger.info("Start process ...")
    if not os.path.isdir(opts.in_path):
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):
            out_path = \
                    os.path.join(opts.out_path,os.path.basename(opts.in_path))
        else:
            out_path = opts.out_path

        evaluater.ffwd_to_img(opts.in_path, out_path, opts.checkpoint_dir, device=opts.device)
    else:
        files = list_files(opts.in_path)
        full_in = [os.path.join(opts.in_path,x) for x in files]
        full_out = [os.path.join(opts.out_path,x) for x in files]
        if opts.allow_different_dimensions:
            evaluater.ffwd_different_dimensions(full_in, full_out, opts.checkpoint_dir,
                    device_t=opts.device, batch_size=opts.batch_size)
        else :
            evaluater.ffwd(full_in, full_out, opts.checkpoint_dir, device_t=opts.device, batch_size=opts.batch_size)
    logger.info("Well Done!")

if __name__ == '__main__':
    main()
