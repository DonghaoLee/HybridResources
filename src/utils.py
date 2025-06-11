# -*- coding utf-8 -*-
# utils.py

# tools for coding

import numpy as np
import logging

def RandomSimplexVector(d=5, size=[1,] ):
    vec = np.random.exponential(scale=1.0, size=size + [d,])
    vec = vec / np.sum(vec, axis=-1).reshape(size + [1,])
    return vec

def compute_avg(e):
    cumsum_e = np.cumsum(e)        
    t = np.arange(1, len(e) + 1)   
    a = cumsum_e / t               
    return a

log_level=logging.INFO    
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=log_level,
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='runlog.log'
)
logger = logging.getLogger('eval')