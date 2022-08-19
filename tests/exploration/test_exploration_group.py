import os, textwrap
import numpy as np
import unittest

from typing import Set, List
from pathlib import Path
try:
    from exploration.context import dpgen2
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dpgen2.exploration.task import (
    NPTTaskGroup, 
    ExplorationStage,
    ReviseTaskGroup,
)
from dpgen2.constants import lmp_conf_name, lmp_input_name, plm_input_name
from unittest.mock import Mock, patch

in_template_npt = textwrap.dedent("""variable        NSTEPS          equal 1000
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal %f
variable        PRES            equal %f
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box          tilt large
if "${restart} > 0" then "read_restart dpgen.restart.*" else "read_data conf.lmp"
change_box   all triclinic
mass            1 10.000000
mass            2 20.000000
pair_style      deepmd model.000.pb model.001.pb model.002.pb  out_freq ${THERMO_FREQ} out_file model_devi.out 
pair_coeff      * *

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}
dump            1 all custom ${DUMP_FREQ} traj.dump id type x y z fx fy fz
restart         10000 dpgen.restart

if "${restart} == 0" then "velocity        all create ${TEMP} 1111"
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}

timestep        0.001000
run             ${NSTEPS} upto
""")

in_template_nvt = textwrap.dedent("""variable        NSTEPS          equal 1000
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal %f
variable        TAU_T           equal 0.100000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box          tilt large
if "${restart} > 0" then "read_restart dpgen.restart.*" else "read_data conf.lmp"
change_box   all triclinic
mass            1 10.000000
mass            2 20.000000
pair_style      deepmd model.000.pb model.001.pb model.002.pb  out_freq ${THERMO_FREQ} out_file model_devi.out 
pair_coeff      * *

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}
dump            1 all custom ${DUMP_FREQ} traj.dump id type x y z fx fy fz
restart         10000 dpgen.restart

if "${restart} == 0" then "velocity        all create ${TEMP} 1111"
fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}

timestep        0.001000
run             ${NSTEPS} upto
""")

in_template_plm = textwrap.dedent("""variable        NSTEPS          equal %f
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal %f
variable        PRES            equal %f
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000

#Initialization
units           metal
dimension       3
atom_style      atomic

read_data       conf.lmp
mass            1 12.011
mass            2  1.008

#Interatomic potentials - DeepMD
pair_style      deepmd model.000.pb model.001.pb model.002.pb  out_freq ${THERMO_FREQ} out_file model_devi.out 
pair_coeff      * *

#MD parameters
timestep        0.0005 #ps
velocity       all create ${TEMP} 1815191 mom yes rot yes dist gaussian

#Run MD - equil at 300K
run_style      verlet #Velocity verlet
fix            dpgen_plm all plumed plumedfile input.plumed outfile output.plumed
fix            1 all nvt temp ${TEMP} ${TEMP} 0.1 #NH thermostat - 300K with 100 fs frequency
fix            2 all momentum 1 linear 0 0 0 #Remove total linear momentum of the system at each step
fix            3 all recenter INIT INIT INIT
thermo_style   custom step temp pe etotal press #Setting printing
thermo         ${THERMO_FREQ} #Ouputing thermodynamic properties
dump           dpgen_dump all custom ${DUMP_FREQ} traj/*.lammpstrj id type x y z\n
#dump          2 all custom 100 vel.xyz id type vx vy vz
run            ${NSTEPS} #25 ps
""")


def swap_element(arg):
    bk = arg.copy()
    arg[1] = bk[0]
    arg[0] = bk[1]

class TestCPTGroup(unittest.TestCase):
    # def setUp(self):
    #     self.mock_random = Mock()

    @patch('dpgen2.exploration.task.lmp.lmp_input.random')
    def test_npt(self, mock_random):
        mock_random.randrange.return_value = 1110
        self.confs = ['foo', 'bar']
        self.tt = [100, 200]
        self.pp = [1, 10, 100]
        self.numb_model = 3
        self.mass_map = [10, 20]

        cpt_group = NPTTaskGroup()
        cpt_group.set_md(
            self.numb_model, 
            self.mass_map,
            self.tt,
            self.pp,
        )
        cpt_group.set_conf(
            self.confs,
        )
        task_group = cpt_group.make_task()

        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs) * len(self.tt) * len(self.pp))
        for ii in range(ngroup):
            i_idx = ii // (len(self.tt) * len(self.pp))
            j_idx = (ii - len(self.tt) * len(self.pp) * i_idx) // len(self.pp)
            k_idx = (ii - len(self.tt) * len(self.pp) * i_idx - len(self.pp) * j_idx)
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name], 
                self.confs[i_idx],
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name], 
                in_template_npt % (self.tt[j_idx], self.pp[k_idx]),
            )

    @patch('dpgen2.exploration.task.lmp.lmp_input.random')
    def test_nvt(self, mock_random):
        mock_random.randrange.return_value = 1110
        self.confs = ['foo', 'bar']
        self.tt = [100, 200]
        self.numb_model = 3
        self.mass_map = [10, 20]

        cpt_group = NPTTaskGroup()
        cpt_group.set_md(
            self.numb_model, 
            self.mass_map,
            self.tt,
            ens = 'nvt',
        )
        cpt_group.set_conf(
            self.confs,
        )
        task_group = cpt_group.make_task()

        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name], 
                self.confs[i_idx],
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name], 
                in_template_nvt % (self.tt[j_idx]),
            )


    @patch('dpgen2.exploration.task.lmp.lmp_input.random')
    def test_nvt_sample(self, mock_random):
        mock_random.randrange.return_value = 1110
        self.confs = ['foo', 'bar']
        self.tt = [100, 200]
        self.numb_model = 3
        self.mass_map = [10, 20]

        cpt_group = NPTTaskGroup()
        cpt_group.set_md(
            self.numb_model, 
            self.mass_map,
            self.tt,
            ens = 'nvt',
        )
        cpt_group.set_conf(
            self.confs,
            n_sample = 1,
        )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name], 
                'foo',
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name], 
                in_template_nvt % (self.tt[j_idx]),
            )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name], 
                'bar',
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name], 
                in_template_nvt % (self.tt[j_idx]),
            )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name], 
                'foo',
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name], 
                in_template_nvt % (self.tt[j_idx]),
            )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name], 
                'bar',
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name], 
                in_template_nvt % (self.tt[j_idx]),
            )


    @patch('dpgen2.exploration.task.npt_task_group.random.shuffle')
    @patch('dpgen2.exploration.task.lmp.lmp_input.random.randrange')
    def test_nvt_sample_random(self, mock_randrange, mock_shuffle):
        mock_randrange.return_value = 1110
        mock_shuffle.side_effect = swap_element
        self.confs = ['foo', 'bar']
        self.tt = [100, 200]
        self.numb_model = 3
        self.mass_map = [10, 20]

        cpt_group = NPTTaskGroup()
        cpt_group.set_md(
            self.numb_model, 
            self.mass_map,
            self.tt,
            ens = 'nvt',
        )
        cpt_group.set_conf(
            self.confs,
            n_sample = 1,
            random_sample = True,
        )            

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name], 
                'bar',
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name], 
                in_template_nvt % (self.tt[j_idx]),
            )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name], 
                'foo',
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name], 
                in_template_nvt % (self.tt[j_idx]),
            )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name], 
                'bar',
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name], 
                in_template_nvt % (self.tt[j_idx]),
            )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name], 
                'foo',
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name], 
                in_template_nvt % (self.tt[j_idx]),
            )

class TestPLMGroup(unittest.TestCase):

    @patch('dpgen2.exploration.task.lmp.lmp_input.random')
    def test_revise(self,mock_random):
        mock_random.randrange.return_value = 1110
        self.conf = ['foo', 'bar']
        self.tt = [100, 200]
        self.ttp = [100, 200]
        self.stride = 10,
        self.numb_model = 3
        self.mass_map = [10, 20]

        lmp_template_path="../../examples/lmp_enhance_sampling/input/input.lammps",
        plm_template_path="../../examples/lmp_enhance_sampling/input/input.plumed"

        lmp_group = ReviseTaskGroup()
        lmp_group.set_md(
            self.numb_model,
            self.mass_map,
            self.tt,
            self.pp,
            self.nsteps,
        )
        task_group_1 = lmp_group.make_task()

        plm_group = ReviseTaskGroup()
        plm_group.set_md(
            self.numb_model,
            self.mass_map,
            self.tt,
            self.pp,
            self.nsteps,
        )
        plm_group.set_plm(
            self.ttp,
            self.stride,
        )
        task_group_2 = plm_group.make_plm_task() 

        ngroup_1 = len(task_group_1)
        self.assertEqual(ngroup_1, len(self.confs) * len(self.tt) * len(self.pp))

        ngroup_2 = len(task_group_2)
        self.assertEqual(ngroup_2, len(self.confs) * len(self.tt) * len(self.ttp) * len(self.pp))

        for ii in range(ngroup_1):
            i_idx = ii // (len(self.tt) * len(self.pp))
            j_idx = (ii - len(self.tt) * len(self.pp) * i_idx) // len(self.pp)
            k_idx = (ii - len(self.tt) * len(self.pp) * i_idx - len(self.pp) * j_idx)
            self.assertEqual(
                task_group_1[ii].files()[lmp_conf_name],
                self.confs[i_idx],
            )
            self.assertEqual(
                task_group_1[ii].files()[lmp_input_name], 
                in_template_npt % (self.tt[j_idx], self.pp[k_idx]),
            )

class TestCPTStage(unittest.TestCase):
    # def setUp(self):
    #     self.mock_random = Mock()

    @patch('dpgen2.exploration.task.lmp.lmp_input.random')
    def test(self, mock_random):
        mock_random.randrange.return_value = 1110
        self.numb_model = 3
        self.mass_map = [10, 20]

        cpt_group_p = NPTTaskGroup()
        cpt_group_p.set_md(
            self.numb_model,
            self.mass_map,
            [100.],
            [1., 10.],
        )
        cpt_group_p.set_conf(
            ['foo'],
        )

        cpt_group_t = NPTTaskGroup()
        cpt_group_t.set_md(
            self.numb_model, 
            self.mass_map,
            [200., 300.],
            ens = 'nvt',
        )
        cpt_group_t.set_conf(
            ['bar'],
        )

        stage = ExplorationStage()
        stage.add_task_group(cpt_group_p).add_task_group(cpt_group_t)

        task_group = stage.make_task()
        
        ngroup = len(task_group)
        self.assertEqual(ngroup, 4)

        ii = 0
        self.assertEqual(task_group[ii].files()[lmp_conf_name], 'foo')
        self.assertEqual(
            task_group[ii].files()[lmp_input_name], 
            in_template_npt % (100., 1.),
        )
        ii+=1
        self.assertEqual(task_group[ii].files()[lmp_conf_name], 'foo')
        self.assertEqual(
            task_group[ii].files()[lmp_input_name], 
            in_template_npt % (100., 10.),
        )
        ii+=1
        self.assertEqual(task_group[ii].files()[lmp_conf_name], 'bar')
        self.assertEqual(
            task_group[ii].files()[lmp_input_name], 
            in_template_nvt % (200.),
        )
        ii+=1
        self.assertEqual(task_group[ii].files()[lmp_conf_name], 'bar')
        self.assertEqual(
            task_group[ii].files()[lmp_input_name], 
            in_template_nvt % (300.),
        )
    
