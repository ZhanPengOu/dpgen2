import random
import numpy as np
from typing import (
    List,
)
from distutils.version import (
    LooseVersion,
)

def find_key(lmp_lines, key):
    found = []
    for idx in range(len(lmp_lines)):
        words = lmp_lines[idx].split()
        nkey = len(key)
        if len(words) >= nkey and words[:nkey] == key:
            found.append(idx)
    return found[0]

def revise_lmp_input_model(lmp_lines, model, trj_freq):
    idx = find_key(lmp_lines, ['pair_style', 'deepmd'])
    graph_list = ' '.join(model)
    lmp_lines[idx] = "pair_style      deepmd %s out_freq ${THERMO_FREQ} out_file model_devi.out\n" % graph_list
    return lmp_lines

def revise_lmp_input_dump(lmp_lines):
    idx = find_key(lmp_lines, ['dump', 'dpgen_dump'])
    lmp_lines[idx] = "dump            dpgen_dump all custom ${DUMP} traj/*.lammpstrj id type x y z\n"

def revise_lmp_input_plm(lmp_lines):
    idx = find_key(lmp_lines, ['fix','dpgen_plm'])
    lmp_lines[idx] = "fix            dpgen_plm all plumed plumedfile input.plumed outfile output.plumed\n"


def make_lmp_template_input(
        conf_file : str,
        nsteps : int,
        graphs: List[str],
        temp : float, 
        pres : float= None,
        
) :
    ## start to revise, first to read the origin lmp template
    with open(lmp_template_path) as fp:   
        lmp_lines = fp.readlines()
        
    ##  revise Temp/Pres/Nsteps key in the template
    rev_dict = dict([('V_TEMP', temp), ('V_PRES', pres), ("V_NSTEPS", nsteps)])
    for key in rev_dict:
        for ii in range(len(lmp_lines)):
            lmp_lines[ii] = lmp_lines[ii].replace(key,str(rev_dict[key]))
    
    ## revise pair_coeff part
    graph_list = ""
    for ii in graph:
        graph_list += ii + " "
    lmp_lines = revise_lmp_input_model(lmp_lines, graph_list) 

    ## revise dump part
    lmp_lines = revise_lmp_input_dump(lmp_lines)

    ## if "model_devi_plumed" is on, revise the plumed part in template
    if(use_plm):
        revise_lmp_input_plm(lmp_lines)

    ret = lmp_lines
    return ret     


