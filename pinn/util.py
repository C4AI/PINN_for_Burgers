import sys
import logging
from typing import Callable
from functools import wraps
import time


def get_log(log_name: str = 'root') -> logging.Logger:
    """
    Sets up logging configs
    """
    new_log = logging.getLogger(log_name)
    log_format = '%(asctime)-24s %(process)-2.2s %(threadName)-8.30s '\
                 '%(levelname)-8s %(name)10s:%(lineno)-4s- %(funcName)s: '\
                 '%(message)s'

    logging.basicConfig(
        stream=sys.stdout,
        format=log_format,
        level=logging.INFO
    )

    return new_log


def perf(fn: Callable):
    '''
    Performance function fn with decorators
    '''
    name = fn.__name__

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        log.info(f'started method {name}')
        ret = fn(*args, **kwargs)
        elapsed = time.time() - start
        log.info('{} took {:.4f}s'.format(name, elapsed))
        return ret
    return wrapper

def initialize_fourier_feature_network(n_transforms,sigma):
    B = {}
    B['t'] = np.array([2*np.pi*i for i in range(1,n_transforms[0]+1)])
    B['x'] = np.random.normal(scale=2*np.pi*sigma[1],size=n_transforms[1])
    B['y'] = np.random.normal(scale=2*np.pi*sigma[2],size=n_transforms[2])

    return B

log = get_log()
