train_index_pattern = "%04d"
train_task_pattern = 'task.' + train_index_pattern
train_script_name = 'input.json'
train_log_name = 'train.log'
model_name_pattern = 'model.%03d.pb'
lmp_index_pattern = "%06d"
lmp_task_pattern = 'task.' + lmp_index_pattern
lmp_conf_name = 'conf.lmp'
lmp_input_name = 'in.lammps'
lmp_traj_name = 'traj.dump'
lmp_log_name = 'log.lammps'
lmp_model_devi_name = 'model_devi.out'
vasp_index_pattern = '%06d'
vasp_task_pattern = 'task.' + vasp_index_pattern
vasp_conf_name = 'POSCAR'
vasp_input_name = 'INCAR'
vasp_pot_name = 'POTCAR'
vasp_kp_name = 'KPOINTS'
vasp_default_log_name = 'vasp.log'
vasp_default_out_data_name = 'data'

default_image = 'dptechnology/dpgen2:latest'
default_host = '127.0.0.1:2746'
