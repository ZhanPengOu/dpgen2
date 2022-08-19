import random
import numpy as np
from typing import (
    List,
)
from distutils.version import (
    LooseVersion,
)

def find_key(plm_lines, key):
    found = []
    for idx in range(len(plm_lines)):
        words = plm_lines[idx].split()
        nkey = len(key)
        if len(words) >= nkey and words[:nkey] == key:
            found.append(idx)
    return found[0]


def make_plm_template_input(
        temp : float, 
        stride: float,
) :
    ## start to revise, first to read the origin lmp template
    with open(plm_template_path) as fp:   
        plm_lines = fp.readlines()
        
    ##  revise Temp/Pres/Nsteps key in the template
    rev_dict = dict([('V_TEMP', temp), ("V_STRIDE", stride)])
    for key in rev_dict:
        for ii in range(len(plm_lines)):
            plm_lines[ii] = plm_lines[ii].replace(key,str(rev_dict[key]))
    
    ret = plm_lines
    return ret     


