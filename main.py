"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.
Description :
Author：Team Li
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from trainers.base_train import Trainer
from utils.config import get_train_config
from utils.logger import LoggerRecord
from utils.utils import *
from trainers.base_evaluate import Evaluater

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        parser = get_train_config()
        options = parser.parse_args()
        check_opts(options)
    except:
        print("missing or invalid arguments")
        exit(0)

    # mkdirs(os.path.join('./', options.style.split('/')[-1][:-4]))
    logName = options.style.split('/')[-1][:-4] + '_cw_' + str(options.content_weight) + '_sw_' + str(options.style_weight)
    logR = LoggerRecord(logName + '.txt', level='info')
    loggerR = logR.logger

    style_target = get_img(options.style)
    content_targets = get_files(options.train_path)

    kwargs = {
        "epochs": options.epochs,
        "print_iterations": options.checkpoint_iterations,
        "batch_size": options.batch_size,
        "save_path": os.path.join(options.checkpoint_dir, 'fns.ckpt'),
        "learning_rate": options.learning_rate,
        'logger': logger,
    }
    args = [
        content_targets,
        style_target,
        options.content_weight,
        options.style_weight,
        options.tv_weight,
        options.vgg_path
    ]

    trainer = Trainer(*args, **kwargs)
    evaluater = Evaluater()
    for preds, losses, i, epoch in trainer.train():
        style_loss, content_loss, tv_loss, loss = losses

        logger.info('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
        to_print = (style_loss, content_loss, tv_loss)
        logger.info('style: %s, content:%s, tv: %s' % to_print)
        if options.test:
            assert options.test_dir != False
            preds_path = '%s/%s_%s.png' % (options.test_dir,epoch,i)
            evaluater.ffwd_to_img(options.test, preds_path, options.checkpoint_dir)


if __name__ == '__main__':
    main()
