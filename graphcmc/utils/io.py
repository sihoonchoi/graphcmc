import os
import numpy as np

def load_atoms_and_charges(framework, ads, charge):
    from ase.io import read
    from importlib.resources import files
    from graphcmc import data

    frame_path = os.path.join(os.getcwd(), f'{framework}.cif')
    framework_atoms = read(frame_path)
    ads_atoms = read(files(data).joinpath(f'{ads}.xyz'))

    if charge:
        with open(frame_path, 'r') as f:
            lines = f.readlines()
        if ' _atom_site_charge\n' in lines:
            idx = lines.index(' _atom_site_charge\n') + 1
            charges = [float(line.strip().split()[-1]) for line in lines[idx:]]
            framework_atoms.set_initial_charges(charges)

        charge_map = {
            'co2': [0.70, -0.35, -0.35],
            'methane': [0.0],
            'h2o': [0.0, 0.241, 0.241, -0.241, -0.241],
        }
        if ads in charge_map:
            ads_atoms.set_initial_charges(charge_map[ads])

    return framework_atoms, ads_atoms

def setup_result_directory(result_dir, job_id, initialize = False, continue_sim = False):
    import shutil
    result_dir = os.path.join(os.getcwd(), result_dir, job_id)

    if initialize and not continue_sim:
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)

    if not initialize and not continue_sim:
        if os.path.exists(os.path.join(result_dir, 'stats.npy')):
            os.remove(os.path.join(result_dir, 'stats.npy'))
        if os.path.exists(os.path.join(result_dir, 'checkpoint.traj')):
            os.remove(os.path.join(result_dir, 'checkpoint.traj'))
        if os.path.exists(os.path.join(result_dir, 'trajectories')):
            shutil.rmtree(os.path.join(result_dir, 'trajectories'))

    os.makedirs(result_dir, exist_ok = True)
    return result_dir

def save_checkpoint(result_dir, stats, molecule_list, iteration, initialize = False):    
    tag = 'initialization_' if initialize else ''

    from ase.io import Trajectory
    with Trajectory(os.path.join(result_dir, f"{tag}checkpoint.traj"), mode = 'w') as traj:
        for atoms in molecule_list:
            traj.write(atoms)

    if not initialize:
        np.save(os.path.join(result_dir, "stats.npy"), stats)

        os.makedirs(os.path.join(result_dir, "trajectories"), exist_ok = True)
        with Trajectory(os.path.join(result_dir, "trajectories", f"movies_{iteration + 1:008d}.traj"), mode = 'w') as traj:
            for atoms in molecule_list:
                traj.write(atoms)

def load_checkpoint(result_dir, initialize = False, continue_sim = False):
    tag = '' if (not initialize and continue_sim) else 'initialization_'

    if initialize and not continue_sim:
        molecule_list = []
    else:
        from ase.io import Trajectory
        traj_path = os.path.join(result_dir, f"{tag}checkpoint.traj")
        if not os.path.exists(traj_path):
            raise FileNotFoundError(f"Trajectory file does not exist: {traj_path}")

        with Trajectory(traj_path) as traj:
            molecule_list = [atoms for atoms in traj]

    stats = {
        'attempted': [0] * 4,
        'accepted': [0] * 4,
        'uptake': [],
        'adsorption_energy': []
    }

    if continue_sim:
        stats_path = os.path.join(result_dir, "stats.npy")
        if not initialize:
            if not os.path.exists(stats_path):
                raise FileNotFoundError(f"Stats file does not exist: {stats_path}")
            stats = np.load(stats_path, allow_pickle = True).item()
        else:
            if os.path.exists(stats_path):
                raise NotImplementedError("The previous calculation was aborted during the production cycles! Cannot continue the initialization cycles.")
  
    return stats, molecule_list
