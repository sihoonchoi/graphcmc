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

def setup_result_directory(job_id, continue_sim):
    result_dir = os.path.join(os.getcwd(), "results", job_id)
    if not continue_sim:
        if os.path.exists(result_dir):
            for f in os.listdir(result_dir):
                os.remove(os.path.join(result_dir, f))
        else:
            os.makedirs(result_dir)
    return result_dir

def save_checkpoint(result_dir, uptake, adsorption_energy, molecule_list, iteration = None, initialize = False):
    tag = 'initialization_' if initialize else ''
    np.save(os.path.join(result_dir, f"{tag}uptake.npy"), np.array(uptake))
    np.save(os.path.join(result_dir, f"{tag}adsorption_energy.npy"), np.array(adsorption_energy))
    np.save(os.path.join(result_dir, "last_adsorbate_positions.npy"), molecule_list)

    if iteration is not None and not initialize:
        np.save(os.path.join(result_dir, f"adsorbate_{iteration + 1:010d}.npy"), molecule_list)

def load_simulation_state(result_dir, initialize):
    tag = 'initialization_' if initialize else ''
    uptake = list(np.load(os.path.join(result_dir, f"{tag}uptake.npy")))
    adsorption_energy = list(np.load(os.path.join(result_dir, f"{tag}adsorption_energy.npy")))
    molecule_list = np.load(os.path.join(result_dir, "last_adsorbate_positions.npy"))

    return uptake, len(uptake), adsorption_energy, molecule_list

def load_stats(result_dir):
    def _load(filename):
        path = os.path.join(result_dir, filename)
        return list(np.load(path))

    accepted = _load("accepted.npy")
    attempted = _load("attempted.npy")

    return accepted, attempted

def save_stats(result_dir, accepted, attempted):
    np.save(os.path.join(result_dir, "accepted.npy"), np.array(accepted))
    np.save(os.path.join(result_dir, "attempted.npy"), np.array(attempted))
