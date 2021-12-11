"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.
Description :
Author：Team Li
"""

import logging
from logging import handlers

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

class LoggerRecord(object):
    # fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
                # S: second
        # M: month
        # H: hour
        # D: day
        # W: weel（interval==0 denotes Monday）
        # midnight
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)

"""
usage 
"""
if __name__ == '__main__':
    best_test_loss = 1.0
    logger.info('Best mean Loss-%.5f' % best_test_loss)

    logR = LoggerRecord('all.txt',level='debug')
    loggerR = logR.logger
    loggerR.info('info')
    logR.logger.debug('debug')
    logR.logger.info('info')
    logR.logger.warning('warning')
    logR.logger.error('error')
    logR.logger.critical('critical')
    LoggerRecord('error.txt', level='error').logger.error('error')
